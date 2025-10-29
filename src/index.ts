/// <reference types="@cloudflare/workers-types" />
import jwt from 'jsonwebtoken';
import * as bcrypt from 'bcryptjs';

export interface Env {
	ISSUER_SECRET: string;
	GROQ_API_KEY: string;
	CHAT_HISTORY_BAYMAX_PROXY: KVNamespace;
	VECTOR_API_URL: string;
	BAYMAX_USERS: KVNamespace;
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
const TOKEN_EXPIRY = 60 * 60;
const TOKEN_REFRESH_THRESHOLD = 60;
const ALLOWED_ORIGINS = ['https://baymax.onslaught2342.qzz.io'];

const LOGIN_ATTEMPT_LIMIT = 5;
const LOGIN_BLOCK_TTL = 60 * 15;
const JSON_BODY_LIMIT = 64 * 1024;

function corsHeaders(origin?: string) {
	const allowOrigin = origin && ALLOWED_ORIGINS.includes(origin) ? origin : '*';
	return {
		'Access-Control-Allow-Origin': allowOrigin,
		'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
		'Access-Control-Allow-Headers': 'Content-Type, Authorization',
	};
}

function jsonResponse(data: unknown, status = 200, request?: Request) {
	const origin = request?.headers.get('Origin') || '*';
	return new Response(JSON.stringify(data), {
		status,
		headers: { 'Content-Type': 'application/json', ...corsHeaders(origin) },
	});
}

function createToken(username: string, env: Env) {
	return jwt.sign({ username }, env.ISSUER_SECRET, { expiresIn: TOKEN_EXPIRY });
}

async function verifyJWT(token: string, env: Env) {
	try {
		const decoded = jwt.verify(token, env.ISSUER_SECRET) as { username?: string; exp?: number };
		if (!decoded || !decoded.username) throw new Error('Invalid token payload');
		return decoded.username;
	} catch (e) {
		throw new Response(JSON.stringify({ error: 'Unauthorized' }), { status: 401, headers: { 'Content-Type': 'application/json' } });
	}
}

async function parseJsonLimited(request: Request) {
	const clone = request.clone();
	const contentLength = request.headers.get('content-length');
	if (contentLength && Number(contentLength) > JSON_BODY_LIMIT) throw new Error('Payload too large');
	const text = await clone.text();
	if (text.length > JSON_BODY_LIMIT) throw new Error('Payload too large');
	try {
		return text ? JSON.parse(text) : null;
	} catch {
		return null;
	}
}

function isValidUsername(u: any) {
	return typeof u === 'string' && /^[a-zA-Z0-9_\-]{3,64}$/.test(u);
}

function isValidPassword(p: any) {
	return typeof p === 'string' && p.length >= 8 && p.length <= 256;
}

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeout = 8000) {
	const controller = new AbortController();
	const id = setTimeout(() => controller.abort(), timeout);
	try {
		const res = await fetch(input, { ...init, signal: controller.signal });
		return res;
	} finally {
		clearTimeout(id);
	}
}

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		try {
			const origin = request.headers.get('Origin') || '';

			if (request.method === 'OPTIONS') {
				return new Response(null, { status: 204, headers: corsHeaders(origin) });
			}

			const url = new URL(request.url);
			const path = url.pathname;

			if (!ALLOWED_ORIGINS.includes(origin)) {
				return jsonResponse({ error: 'Forbidden origin' }, 403, request);
			}

			if (path === '/signup' && request.method === 'POST') {
				const body = await parseJsonLimited(request).catch(() => null);
				if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400, request);

				const { username, password } = body;
				if (!isValidUsername(username) || !isValidPassword(password)) {
					return jsonResponse({ error: 'Invalid username or password format' }, 400, request);
				}

				const exists = await env.BAYMAX_USERS.get(username);
				if (exists) return jsonResponse({ error: 'Username already exists' }, 409, request);

				const hashed = await bcrypt.hash(password, 10);
				await env.BAYMAX_USERS.put(username, hashed);

				const token = createToken(username, env);
				await env.BAYMAX_USERS.put(`${username}_token`, token);

				return jsonResponse({ token }, 200, request);
			}

			if (path === '/login' && request.method === 'POST') {
				const body = await parseJsonLimited(request).catch(() => null);
				if (!body) return jsonResponse({ error: 'Invalid JSON' }, 400, request);

				const { username, password } = body;
				if (!isValidUsername(username) || !isValidPassword(password)) {
					return jsonResponse({ error: 'Username and password required' }, 400, request);
				}

				const blockKey = `${username}_login_block`;
				const attemptsKey = `${username}_login_attempts`;
				const blocked = await env.BAYMAX_USERS.get(blockKey);
				if (blocked) return jsonResponse({ error: 'Too many login attempts. Try later.' }, 429, request);

				const stored = await env.BAYMAX_USERS.get(username);
				if (!stored) {
					const curAttempts = Number((await env.BAYMAX_USERS.get(attemptsKey)) || 0) + 1;
					await env.BAYMAX_USERS.put(attemptsKey, String(curAttempts), { expirationTtl: LOGIN_BLOCK_TTL });
					if (curAttempts >= LOGIN_ATTEMPT_LIMIT) {
						await env.BAYMAX_USERS.put(blockKey, '1', { expirationTtl: LOGIN_BLOCK_TTL });
						return jsonResponse({ error: 'Too many login attempts. Try later.' }, 429, request);
					}
					return jsonResponse({ error: 'Invalid credentials' }, 401, request);
				}

				const valid = await bcrypt.compare(password, stored);
				if (!valid) {
					const curAttempts = Number((await env.BAYMAX_USERS.get(attemptsKey)) || 0) + 1;
					await env.BAYMAX_USERS.put(attemptsKey, String(curAttempts), { expirationTtl: LOGIN_BLOCK_TTL });
					if (curAttempts >= LOGIN_ATTEMPT_LIMIT) {
						await env.BAYMAX_USERS.put(blockKey, '1', { expirationTtl: LOGIN_BLOCK_TTL });
						return jsonResponse({ error: 'Too many login attempts. Try later.' }, 429, request);
					}
					return jsonResponse({ error: 'Invalid credentials' }, 401, request);
				}

				await env.BAYMAX_USERS.delete(attemptsKey);
				await env.BAYMAX_USERS.delete(blockKey);

				const token = createToken(username, env);
				await env.BAYMAX_USERS.put(`${username}_token`, token);

				return jsonResponse({ token }, 200, request);
			}

			if (path === '/verify') {
				const authHeader = request.headers.get('Authorization') || '';
				const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
				if (!token) return jsonResponse({ error: 'Missing token' }, 401, request);

				const username = await verifyJWT(token, env);
				return jsonResponse({ username }, 200, request);
			}

			if (request.method !== 'POST') return jsonResponse({ error: 'Only POST allowed' }, 405, request);

			const authHeader = request.headers.get('Authorization') || '';
			const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : null;
			if (!token) return jsonResponse({ error: 'Unauthorized: missing token' }, 401, request);

			const username = await verifyJWT(token, env);

			let refreshToken: string | null = null;
			const decoded: any = jwt.decode(token);
			if (decoded && decoded.exp && Date.now() / 1000 + TOKEN_REFRESH_THRESHOLD > decoded.exp) {
				refreshToken = createToken(username, env);
				await env.BAYMAX_USERS.put(`${username}_token`, refreshToken);
			}

			const body = await parseJsonLimited(request).catch(() => null);
			if (!body) return jsonResponse({ error: 'Invalid JSON', refreshToken }, 400, request);

			const userMessage = typeof body.userMessage === 'string' ? body.userMessage.trim() : null;
			if (!userMessage) return jsonResponse({ error: 'Missing userMessage', refreshToken }, 400, request);
			if (userMessage.length > 20000) return jsonResponse({ error: 'userMessage too long', refreshToken }, 413, request);

			const sessionId = body.sessionId;
			if (!sessionId || typeof sessionId !== 'string' || sessionId.length > 256) {
				return jsonResponse({ error: 'Missing or invalid sessionId', refreshToken }, 400, request);
			}

			const sanitizedInput = userMessage.replace(/\s+/g, ' ').trim();
			const historyRaw = await env.CHAT_HISTORY_BAYMAX_PROXY.get(sessionId);
			let history: Array<{ role: string; content: string }> = historyRaw
				? JSON.parse(historyRaw)
				: [{ role: 'system', content: 'You are Baymax, a friendly medical AI giving safe health advice.' }];

			history.push({ role: 'user', content: sanitizedInput });
			const systemMessage = history.find((m) => m.role === 'system') ?? null;
			const nonSystem = history.filter((m) => m.role !== 'system').slice(-(MAX_HISTORY - (systemMessage ? 1 : 0)));
			history = systemMessage ? [systemMessage, ...nonSystem] : nonSystem;

			let context = '';
			let embeddings: any[] = [];
			try {
				const vectorRes = await fetchWithTimeout(
					env.VECTOR_API_URL,
					{
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({ query: sanitizedInput }),
					},
					4000
				);
				if (vectorRes && vectorRes.ok) {
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
				}
			} catch (err) {
				context = '';
			}

			if (context) {
				history.unshift({
					role: 'system',
					content: `Medical database context:\n${context}\n\nRespond as Baymax, using this data carefully.`,
				});
			}

			const groqPayload = { model: 'llama-3.1-8b-instant', messages: history, temperature: 0.7 };
			const startTime = Date.now();

			const groqRes = await fetchWithTimeout(
				'https://api.groq.com/openai/v1/chat/completions',
				{
					method: 'POST',
					headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.GROQ_API_KEY}` },
					body: JSON.stringify(groqPayload),
				},
				12000
			);
			const latency = Date.now() - startTime;

			if (!groqRes || !groqRes.ok) {
				const errText = groqRes ? await groqRes.text().catch(() => '<no-body>') : '<no-response>';
				return jsonResponse(
					{ error: 'Upstream model error', upstreamStatus: groqRes?.status ?? 0, detail: errText, refreshToken },
					502,
					request
				);
			}

			const data = (await groqRes.json().catch(() => null)) as GroqResponse | null;
			const rawReply =
				data?.choices?.[0]?.message?.content ??
				data?.choices?.[0]?.text ??
				(typeof data?.reply === 'string' ? data.reply : null) ??
				'Model returned no reply';
			const refinedReply = typeof rawReply === 'string' ? rawReply.trim() : String(rawReply);

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
				request
			);
		} catch (err: any) {
			if (err instanceof Response) return err;
			console.error('Unhandled worker error:', err);
			return jsonResponse({ error: err.message ?? 'Internal Server Error' }, 500);
		}
	},
};
