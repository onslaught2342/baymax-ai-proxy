/// <reference types="@cloudflare/workers-types" />
import jwt from 'jsonwebtoken';
import * as bcrypt from 'bcryptjs';

export interface Env {
	ISSUER_SECRET: string;
	GROQ_API_KEY: string;
	CHAT_HISTORY_BAYMAX_PROXY: KVNamespace;
	VECTOR_API_URL: string;
	BAYMAX_USERS: KVNamespace; // For auth
}

interface VectorResult {
	'Medicine Name'?: string;
	Uses?: string;
	Side_effects?: string;
}

interface VectorResponse {
	results: VectorResult[];
}

interface GroqChoice {
	message?: { content?: string };
	text?: string;
}

interface GroqResponse {
	choices?: GroqChoice[];
	reply?: string;
}

const MAX_HISTORY = 20;
const HISTORY_TTL = 60 * 60 * 24 * 30;
const TOKEN_EXPIRY = 60 * 60; // 1 hour
const TOKEN_REFRESH_THRESHOLD = 60;

function corsHeaders(origin?: string) {
	const allowOrigin = origin ?? '*';
	return {
		'Access-Control-Allow-Origin': allowOrigin,
		'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
		'Access-Control-Allow-Headers': 'Content-Type, Authorization',
	};
}

function jsonResponse(data: unknown, status = 200, origin?: string) {
	return new Response(JSON.stringify(data), {
		status,
		headers: { 'Content-Type': 'application/json', ...corsHeaders(origin) },
	});
}

function createToken(username: string) {
	return jwt.sign({ username }, ISSUER_SECRET, { expiresIn: TOKEN_EXPIRY });
}

async function verifyJWT(token: string) {
	try {
		const decoded = jwt.verify(token, ISSUER_SECRET) as { username: string };
		if (!decoded.username) throw new Error('Invalid token payload');
		return decoded.username;
	} catch {
		throw new Response('Unauthorized', { status: 401 });
	}
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const origin = request.headers.get('Origin') || '';
		const ALLOWED_ORIGIN = 'https://baymax.onslaught2342.qzz.io';
		if (origin !== ALLOWED_ORIGIN) return new Response('Forbidden', { status: 403 });

		const url = new URL(request.url);
		const path = url.pathname;

		// Preflight
		if (request.method === 'OPTIONS') {
			return new Response(null, {
				status: 204,
				headers: { ...corsHeaders(origin), 'Access-Control-Allow-Credentials': 'false', 'Access-Control-Max-Age': '86400' },
			});
		}

		// Signup
		if (path === '/signup' && request.method === 'POST') {
			let body: any;
			try {
				body = await request.json();
			} catch {
				return jsonResponse({ error: 'Invalid JSON' }, 400);
			}
			const { username, password } = body;
			if (!username || !password) return jsonResponse({ error: 'Username and password required' }, 400);

			const exists = await env.BAYMAX_USERS.get(username);
			if (exists) return jsonResponse({ error: 'Username already exists' }, 409);

			const hashed = await bcrypt.hash(password, 10);
			await env.BAYMAX_USERS.put(username, hashed);

			const token = createToken(username);
			await env.BAYMAX_USERS.put(`${username}_token`, token);
			return jsonResponse({ token });
		}

		// Login
		if (path === '/login' && request.method === 'POST') {
			let body: any;
			try {
				body = await request.json();
			} catch {
				return jsonResponse({ error: 'Invalid JSON' }, 400);
			}
			const { username, password } = body;
			if (!username || !password) return jsonResponse({ error: 'Username and password required' }, 400);

			const stored = await env.BAYMAX_USERS.get(username);
			if (!stored) return jsonResponse({ error: 'Invalid credentials' }, 401);

			const valid = await bcrypt.compare(password, stored);
			if (!valid) return jsonResponse({ error: 'Invalid credentials' }, 401);

			const token = createToken(username);
			await env.BAYMAX_USERS.put(`${username}_token`, token);

			return jsonResponse({ token });
		}

		// Verify token endpoint (optional)
		if (path === '/verify') {
			const authHeader = request.headers.get('Authorization') || '';
			const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
			if (!token) return jsonResponse({ error: 'Missing token' }, 401);
			try {
				const username = await verifyJWT(token);
				return jsonResponse({ username });
			} catch (err) {
				return err instanceof Response ? err : jsonResponse({ error: 'Unauthorized' }, 401);
			}
		}

		// For all other requests (Baymax chat)
		if (request.method !== 'POST') return jsonResponse({ error: 'Only POST allowed' }, 405);

		// Authentication required
		const authHeader = request.headers.get('Authorization') || '';
		const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
		if (!token) return new Response('Unauthorized: missing token', { status: 401 });

		let username: string;
		try {
			username = await verifyJWT(token);
		} catch (err) {
			return err instanceof Response ? err : new Response('Unauthorized', { status: 401 });
		}

		// Optionally refresh token
		let refreshToken: string | null = null;
		const decoded: any = jwt.decode(token);
		if (decoded && decoded.exp && Date.now() / 1000 + TOKEN_REFRESH_THRESHOLD > decoded.exp) {
			refreshToken = createToken(username);
			await env.BAYMAX_USERS.put(`${username}_token`, refreshToken);
		}

		// Chat logic
		let body: any;
		try {
			body = await request.json();
		} catch {
			return jsonResponse({ error: 'Invalid JSON', refreshToken }, 400, origin);
		}
		let userMessage = typeof body.userMessage === 'string' ? body.userMessage.trim() : null;
		if (!userMessage) return jsonResponse({ error: 'Missing userMessage', refreshToken }, 400, origin);
		if (userMessage.length > 20000) return jsonResponse({ error: 'userMessage too long', refreshToken }, 413, origin);

		const sessionId = body.sessionId;
		if (!sessionId) return jsonResponse({ error: 'Missing sessionId', refreshToken }, 400, origin);

		const sanitizedInput = userMessage.replace(/\s+/g, ' ').trim();
		const historyRaw = await env.CHAT_HISTORY_BAYMAX_PROXY.get(sessionId);
		let history: Array<{ role: string; content: string }> = historyRaw
			? JSON.parse(historyRaw)
			: [{ role: 'system', content: 'You are Baymax, a friendly medical AI giving safe health advice.' }];

		history.push({ role: 'user', content: sanitizedInput });
		const systemMessage = history.find((m) => m.role === 'system') ?? null;
		const nonSystem = history.filter((m) => m.role !== 'system').slice(-(MAX_HISTORY - (systemMessage ? 1 : 0)));
		history = systemMessage ? [systemMessage, ...nonSystem] : nonSystem;

		// Vector embedding context
		let context = '';
		let embeddings: any[] = [];
		try {
			const vectorRes = await fetch(env.VECTOR_API_URL, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ query: sanitizedInput }),
			});
			const vectorData = (await vectorRes.json().catch(() => null)) as VectorResponse | null;
			if (vectorData?.results?.length) {
				context = vectorData.results
					.map((r) => {
						const name = r['Medicine Name'] ?? 'Unknown';
						const use = r['Uses'] ?? 'N/A';
						const side = r['Side_effects'] ?? 'None listed';
						return `• ${name}: Used for ${use}. Side effects: ${side}`;
					})
					.join('\n');
				embeddings = vectorData.results;
			}
		} catch {
			context = '';
		}

		if (context)
			history.unshift({
				role: 'system',
				content: `Medical database context:\n${context}\n\nRespond as Baymax, using this data carefully.`,
			});

		const groqPayload = { model: 'llama-3.1-8b-instant', messages: history, temperature: 0.7 };
		const startTime = Date.now();
		const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.GROQ_API_KEY}` },
			body: JSON.stringify(groqPayload),
		});
		const latency = Date.now() - startTime;

		if (!groqRes.ok) {
			const errText = await groqRes.text().catch(() => '<no-body>');
			return jsonResponse({ error: 'Upstream model error', upstreamStatus: groqRes.status, detail: errText, refreshToken }, 502, origin);
		}

		const data = (await groqRes.json().catch(() => null)) as GroqResponse | null;
		const rawReply =
			data?.choices?.[0]?.message?.content ??
			data?.choices?.[0]?.text ??
			(typeof data?.reply === 'string' ? data.reply : null) ??
			'Model returned no reply';
		const refinedReply = rawReply.trim();

		history.push({ role: 'assistant', content: refinedReply });
		await env.CHAT_HISTORY_BAYMAX_PROXY.put(sessionId, JSON.stringify(history), { expirationTtl: HISTORY_TTL });

		return jsonResponse(
			{
				input: { original: userMessage, sanitized: sanitizedInput, embeddings },
				output: { raw: rawReply, refined: refinedReply, choices: data?.choices ?? null },
				nonSensitive: { context, metrics: { latency, historyLength: history.length, hallucinationDetected: false } },
				refreshToken,
			},
			200,
			origin
		);
	},
};
