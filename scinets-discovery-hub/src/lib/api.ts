import type { RunRequest, ExperimentRequest, SSEEvent, ActivityEvent, DiscoveryResult, QuotaInfo } from './types';

const API_URL = import.meta.env.VITE_API_URL;

export type SSECallback = {
    onActivity?: (activity: ActivityEvent) => void;
    onLog?: (message: string) => void;
    onResult?: (result: Partial<DiscoveryResult>) => void;
    onError?: (error: string) => void;
    onInterrupt?: (data: { next: string[]; thread_id: string }) => void;
    onDone?: () => void;
    onThreadId?: (threadId: string) => void;
    onQuota?: (info: QuotaInfo) => void;
};

/**
 * Start a discovery stream via SSE.
 * Returns an AbortController to cancel the stream.
 */
export async function startDiscoveryStream(
    request: RunRequest,
    callbacks: SSECallback
): Promise<AbortController> {
    const controller = new AbortController();

    try {
        const response = await fetch(`${API_URL}/run_stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            body: JSON.stringify(request),
            signal: controller.signal,
            credentials: "include"
        });

        if (!response.ok) {
            let errorMsg = `HTTP ${response.status}: ${response.statusText}`;
            try {
                // Clone response to avoid consuming body if we need it later (though we throw anyway)
                const errorData = await response.clone().json();
                if (errorData.detail) {
                    // Handle Rate Limit Detail
                    if (typeof errorData.detail === 'object') {
                        const d = errorData.detail;
                        if (d.error === 'LIMIT_REACHED') {
                            errorMsg = `Rate limit reached. Unlocks at: ${new Date(d.unlocks_at).toLocaleString()}`;
                        } else if (d.error === 'quota_exceeded') {
                            errorMsg = `Quota Exceeded: ${d.message} (Resets in ${d.resets_in_hours} hours)`;
                        } else {
                            // Fallback for other objects
                            errorMsg = d.message || JSON.stringify(d);
                        }
                    } else {
                        // Handle other detail formats (string)
                        errorMsg = String(errorData.detail);
                    }
                }
            } catch (e) {
                // Ignore json parse error
            }
            throw new Error(errorMsg);
        }

        if (!response.body) {
            throw new Error('No response body');
        }

        // Process SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        const processStream = async () => {
            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    callbacks.onDone?.();
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6).trim();

                        if (data === '[DONE]') {
                            callbacks.onDone?.();
                            return;
                        }

                        try {
                            const event: SSEEvent = JSON.parse(data);

                            // Capture thread_id from first event
                            if (event.thread_id) {
                                callbacks.onThreadId?.(event.thread_id);
                            }

                            switch (event.type) {
                                case 'activity':
                                    callbacks.onActivity?.(event.data as ActivityEvent);
                                    break;
                                case 'log':
                                    callbacks.onLog?.(event.data as string);
                                    break;
                                case 'result':
                                    callbacks.onResult?.(event.data as Partial<DiscoveryResult>);
                                    break;
                                case 'error':
                                    callbacks.onError?.(event.data as string);
                                    break;
                                case 'interrupt':
                                    callbacks.onInterrupt?.(event.data as { next: string[]; thread_id: string });
                                    break;
                                case 'quota':
                                    callbacks.onQuota?.(event.data as QuotaInfo);
                                    break;
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE event:', data, e);
                        }
                    }
                }
            }
        };

        processStream().catch((err) => {
            if (err.name !== 'AbortError') {
                callbacks.onError?.(err.message);
            }
        });

    } catch (err) {
        if ((err as Error).name !== 'AbortError') {
            callbacks.onError?.((err as Error).message);
        }
    }

    return controller;
    return controller;
}

/**
 * Resume a discovery stream with selected hypotheses.
 * This re-uses startDiscoveryStream but ensures thread_id is passed.
 */
export async function resumeDiscoveryStream(
    threadId: string,
    selectedHypothesisIds: string[],
    callbacks: SSECallback
): Promise<AbortController> {
    // Construct a resume request payload
    // We only need thread_id and selection, backend handles state restoration
    const request: RunRequest = {
        query: "", // Will be ignored by backend on resume
        thread_id: threadId,
        selected_hypothesis_ids: selectedHypothesisIds,
        // Default placeholders to satisfy type (backend ignores these on resume)
        goal: "discover",
        timeline: "recent",
        max_papers: 10,
        run_experiments: false
    };

    return startDiscoveryStream(request, callbacks);
}

/**
 * Start an experiment execution stream via SSE.
 */
export async function startExperimentStream(
    request: ExperimentRequest,
    callbacks: SSECallback
): Promise<AbortController> {
    const controller = new AbortController();

    try {
        const response = await fetch(`${API_URL}/experiment_stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
            },
            body: JSON.stringify(request),
            signal: controller.signal,
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        if (!response.body) {
            throw new Error('No response body');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        const processStream = async () => {
            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    callbacks.onDone?.();
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6).trim();

                        if (data === '[DONE]') {
                            callbacks.onDone?.();
                            return;
                        }

                        try {
                            const event: SSEEvent = JSON.parse(data);

                            switch (event.type) {
                                case 'activity':
                                    callbacks.onActivity?.(event.data as ActivityEvent);
                                    break;
                                case 'log':
                                    callbacks.onLog?.(event.data as string);
                                    break;
                                case 'result':
                                    callbacks.onResult?.(event.data as Partial<DiscoveryResult>);
                                    break;
                                case 'error':
                                    callbacks.onError?.(event.data as string);
                                    break;
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE event:', data, e);
                        }
                    }
                }
            }
        };

        processStream().catch((err) => {
            if (err.name !== 'AbortError') {
                callbacks.onError?.(err.message);
            }
        });

    } catch (err) {
        if ((err as Error).name !== 'AbortError') {
            callbacks.onError?.((err as Error).message);
        }
    }

    return controller;
}

/**
 * Health check
 */
export async function checkHealth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        return data.status === 'ok';
    } catch {
        return false;
    }
}
