export interface Env {
	GROQ_API_KEY: string;
	CHAT_HISTORY_BAYMAX_PROXY: KVNamespace;
}

const MAX_HISTORY = 20;
const HISTORY_TTL = 60 * 60 * 24 * 30;
const VECTOR_API_URL = 'https://vector_local.onslaught2342.qzz.io/search'; // your tunnel endpoint

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

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const origin = request.headers.get('Origin') || '*';
		const url = new URL(request.url);

		try {
			if (request.method === 'OPTIONS') {
				return new Response(null, {
					status: 204,
					headers: {
						...corsHeaders(origin),
						'Access-Control-Allow-Credentials': 'false',
						'Access-Control-Max-Age': '86400',
					},
				});
			}

			if (url.pathname === '/clear' && request.method === 'POST') {
				let body: any;
				try {
					body = await request.json();
				} catch {
					return jsonResponse({ error: 'Invalid JSON body' }, 400, origin);
				}

				const sessionId = typeof body.sessionId === 'string' && body.sessionId.trim() ? body.sessionId.trim() : null;

				if (!sessionId) {
					return jsonResponse({ error: 'Missing sessionId' }, 400, origin);
				}

				try {
					await env.CHAT_HISTORY_BAYMAX_PROXY.delete(sessionId);
					return jsonResponse({ success: true, message: 'Session cleared' }, 200, origin);
				} catch (err) {
					const msg = err instanceof Error ? err.message : String(err);
					return jsonResponse({ error: 'Failed to clear session', message: msg }, 500, origin);
				}
			}

			if (request.method !== 'POST') {
				return jsonResponse({ error: 'Only POST allowed' }, 405, origin);
			}

			const contentType = (request.headers.get('Content-Type') || '').toLowerCase();
			if (!contentType.includes('application/json')) {
				return jsonResponse({ error: 'Expected application/json' }, 415, origin);
			}

			let body: any;
			try {
				body = await request.json();
			} catch {
				return jsonResponse({ error: 'Invalid JSON body' }, 400, origin);
			}

			const sessionId = typeof body.sessionId === 'string' && body.sessionId.trim() ? body.sessionId.trim() : null;
			const userMessage = typeof body.userMessage === 'string' ? body.userMessage.trim() : null;

			if (!sessionId || !userMessage) {
				return jsonResponse({ error: 'Missing sessionId or userMessage' }, 400, origin);
			}

			if (userMessage.length > 20000) {
				return jsonResponse({ error: 'userMessage too long' }, 413, origin);
			}

			const historyRaw = await env.CHAT_HISTORY_BAYMAX_PROXY.get(sessionId);
			let history: Array<{ role: string; content: string }> = historyRaw
				? JSON.parse(historyRaw)
				: [{ role: 'system', content: 'You are Baymax, a friendly medical AI giving safe health advice.' }];

			history.push({ role: 'user', content: userMessage });

			const systemMessage = history.find((m) => m.role === 'system') ?? null;
			const nonSystem = history.filter((m) => m.role !== 'system').slice(-(MAX_HISTORY - (systemMessage ? 1 : 0)));
			history = systemMessage ? [systemMessage, ...nonSystem] : nonSystem;
			let context = '';
			try {
				const vectorRes = await fetch(VECTOR_API_URL, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ query: userMessage }),
				});

				const vectorData = await vectorRes.json().catch(() => null);

				if (vectorData?.results?.length) {
					context = vectorData.results
						.map((r: any) => {
							const name = r['Medicine Name'] ?? 'Unknown';
							const use = r['Uses'] ?? 'N/A';
							const side = r['Side_effects'] ?? 'None listed';
							return `• ${name}: Used for ${use}. Side effects: ${side}`;
						})
						.join('\n');
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

			const groqPayload = {
				model: 'llama-3.1-8b-instant',
				messages: history,
				temperature: 0.7,
			};

			const groqRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					Authorization: `Bearer ${env.GROQ_API_KEY}`,
				},
				body: JSON.stringify(groqPayload),
			});

			if (!groqRes.ok) {
				const errText = await groqRes.text().catch(() => '<no-body>');
				return jsonResponse(
					{
						error: 'Upstream model error',
						upstreamStatus: groqRes.status,
						detail: errText,
					},
					502,
					origin
				);
			}

			const data = await groqRes.json().catch(() => ({} as any));
			const reply =
				data?.choices?.[0]?.message?.content ??
				data?.choices?.[0]?.text ??
				(typeof data?.reply === 'string' ? data.reply : null) ??
				'Model returned no reply';

			history.push({ role: 'assistant', content: reply });
			await env.CHAT_HISTORY_BAYMAX_PROXY.put(sessionId, JSON.stringify(history), {
				expirationTtl: HISTORY_TTL,
			});

			return jsonResponse({ reply, context, choices: data?.choices ?? null }, 200, origin);
		} catch (err) {
			const message = err instanceof Error ? err.message : String(err);
			return jsonResponse({ error: 'Internal Server Error', message }, 500, origin);
		}
	},
};
