import axios from 'axios';

const API_BASE = 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
});

export interface Model {
    id: string;
    name: string;
    type?: 'chat' | 'embedding';
    provider?: string;
    context_length: number;
    description?: string;
    pricing: {
        prompt: string;
        completion: string;
        request?: string;
        image?: string;
    };
    supports_tools: boolean;
    supported_parameters?: string[];
}

export interface CognitiveState {
    valence: number;
    arousal: number;
    intrinsic_dimension: number;
    gate_value: number;
    primary_manifold: string;
}

export const api = {
    fetchModels: async (): Promise<Model[]> => {
        const apiKey = localStorage.getItem('OPENROUTER_API_KEY');
        const headers: Record<string, string> = {};
        if (apiKey) {
            headers['Authorization'] = `Bearer ${apiKey}`;
            // headers['X-OpenRouter-Key'] = apiKey;
        }
        try {
            const res = await apiClient.get('/system/models', { headers });
            return res.data;
        } catch (e) {
            console.error("Fetch Models failed:", e);
            return [];
        }
    },

    listModels: async (provider?: string): Promise<Model[]> => {
        const apiKey = localStorage.getItem('OPENROUTER_API_KEY');
        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
        };
        if (apiKey) {
            headers['Authorization'] = `Bearer ${apiKey}`;
        }
        try {
            const res = await apiClient.get('/system/models', { headers, params: { provider } });
            return res.data;
        } catch (e) {
            console.error("Failed to list models", e);
            return [];
        }
    },

    getConfig: async () => {
        const res = await apiClient.get('/system/config');
        return res.data;
    },

    updateConfig: async (config: any) => {
        const res = await apiClient.post('/system/config', config);
        return res.data;
    },

    unloadModel: async (modelName?: string) => {
        const res = await apiClient.post('/system/unload-model', { model: modelName });
        return res.data;
    },

    getSessions: async () => {
        const res = await apiClient.get('/chat/sessions');
        return res.data;
    },

    getHistory: async (sessionId: string) => {
        const res = await apiClient.get(`/chat/history/${sessionId}`);
        return res.data;
    },

    getGraphState: async (sessionId?: string) => {
        try {
            const res = await apiClient.get('/chat/graph', { params: { session_id: sessionId } });
            return res.data;
        } catch (e) {
            console.error("Failed to fetch graph state", e);
            return { nodes: [], links: [] };
        }
    },

    getMcpStatus: async () => {
        try {
            const res = await apiClient.get('/mcp/status');
            return res.data;
        } catch (e) {
            console.error("Failed to fetch MCP status", e);
            return { servers: [], status: "error" };
        }
    },

    getSkills: async () => {
        try {
            const res = await apiClient.get('/skills');
            return res.data;
        } catch (e) {
            console.error("Failed to fetch skills", e);
            return [];
        }
    },

    listSessions: async () => {
        const res = await apiClient.get('/chat/sessions');
        return res.data;
    },

    streamChat: (payload: any, onEvent: (event: any) => void) => {
        const ctrl = new AbortController();

        const apiKey = localStorage.getItem('OPENROUTER_API_KEY');
        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            'HTTP-Referer': 'https://nexus-ai.local', // Required by OpenRouter
            'X-Title': 'NEXUS Cognitive Architecture',
        };

        if (apiKey) {
            headers['Authorization'] = `Bearer ${apiKey}`;
            // headers['X-OpenRouter-Key'] = apiKey; // OpenRouter prefers Auth Bearer standard usually, but X-OpenRouter-Key is also valid. Let's stick to standard if possible, or keep existing if it works. Code says `X-OpenRouter-Key` was used. Let's use both or check docs. OpenRouter docs say "Authorization: Bearer $OPENROUTER_API_KEY".
            headers['X-OpenRouter-Key'] = apiKey; // Keeping for backward compat if any, but adding standard auth header too? No, let's just stick to what was there or standard. The previous code ONLY used X-OpenRouter-Key. I will add Authorization for safety.
        }

        // Use fetch for SSE
        fetch(`${API_BASE}/chat/completions`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                ...payload,
                max_tool_iterations: parseInt(localStorage.getItem('NEXUS_MAX_ITERATIONS') || '50')
            }),
            signal: ctrl.signal
        }).then(async (response) => {
            const reader = response.body?.getReader();
            const decoder = new TextDecoder();

            if (!reader) return;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('event: ')) {
                        // Handle named events
                        const [eventLine, dataLine] = line.split('\n');
                        const eventName = eventLine.replace('event: ', '').trim();
                        const dataStr = dataLine?.replace('data: ', '').trim();
                        if (dataStr) {
                            try {
                                onEvent({ type: eventName, data: JSON.parse(dataStr) });
                            } catch (e) {
                                console.error("Parse error", e);
                            }
                        }
                    } else if (line.startsWith('data: ')) {
                        // Default data event (tokens usually, but now also usage)
                        const dataStr = line.replace('data: ', '').trim();
                        if (dataStr === '[DONE]') {
                            onEvent({ type: 'done' });
                        } else {
                            try {
                                const parsed = JSON.parse(dataStr);
                                if (parsed.type === 'usage') {
                                    onEvent(parsed);
                                } else if (parsed.type === 'thinking') {
                                    onEvent(parsed);
                                } else if (parsed.type === 'tool_call_chunk') {
                                    onEvent(parsed);
                                } else if (parsed.type === 'graph_update') {
                                    onEvent(parsed);
                                } else if (parsed.type === 'code_output') {
                                    onEvent(parsed);
                                } else if (parsed.type === 'code_output_chunk') {
                                    onEvent(parsed);
                                } else {
                                    // Fallback for standard token
                                    onEvent({ type: 'token', ...parsed });
                                }
                            } catch (e) { console.error(e) }
                        }
                    }
                }
            }
        });

        return ctrl;
    }
};
