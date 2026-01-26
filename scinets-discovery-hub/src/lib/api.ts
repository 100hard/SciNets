import type { RunRequest, ExperimentRequest, SSEEvent, ActivityEvent, DiscoveryResult, QuotaInfo } from './types';
import { config } from "../config";

const API_URL = config.API_URL;

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
    let currentThreadId = request.thread_id; // Start with requested ID (if resume)
    let retryCount = 0;
    const MAX_RETRIES = 5;

    // Retry Loop Wrapper
    const connect = async () => {
        try {
            // FIX: If retrying/reconnecting, ensure thread_id is set
            const currentRequest = { ...request };
            if (currentThreadId) {
                currentRequest.thread_id = currentThreadId;
                // currentRequest.is_resume = true; // Not needed, backend infers from thread_id
            }

            const response = await fetch(`${API_URL}/run_stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream',
                },
                body: JSON.stringify(currentRequest),
                signal: controller.signal,
                credentials: "include" // REQUIRED: Send cookies cross-origin
            });

            if (!response.ok) {
                // non-2xx response -> meaningful error (auth, validation), DO NOT RETRY unless 502/503/504
                if ([502, 503, 504].includes(response.status)) {
                    throw new Error(`Gateway Error ${response.status}`); // Throw to trigger retry
                }
                const errorMsg = `HTTP ${response.status}: ${response.statusText}`;
                // ... (Parsing logic omitted for brevity, assume throws) ...
                try {
                    const errorData = await response.clone().json();
                    if (errorData.detail) throw new Error(String(errorData.detail));
                } catch (e) { }
                throw new Error(errorMsg);
            }

            if (!response.body) {
                throw new Error('No response body');
            }

            // Stream connected - Reset retries
            retryCount = 0;

            // Process SSE stream
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            try {
                while (true) {
                    const { done, value } = await reader.read();

                    if (done) {
                        // Stream finished naturally
                        // Check if we actually got [DONE] signal? 
                        // If not, it might be a silent premature close.
                        // Ideally we track "finished" state.
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
                                return; // Success exit
                            }

                            // Heartbeat - Ignore empty data or ping
                            if (data === '{}' || !data) continue;

                            try {
                                const event: SSEEvent = JSON.parse(data);

                                // Capture thread_id from first valid event
                                if (event.thread_id && !currentThreadId) {
                                    currentThreadId = event.thread_id;
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
                                        return; // Logic error from backend -> Stop
                                    case 'interrupt':
                                        callbacks.onInterrupt?.(event.data as { next: string[]; thread_id: string });
                                        return; // Expected stop
                                    case 'quota':
                                        callbacks.onQuota?.(event.data as QuotaInfo);
                                        break;
                                }
                            } catch (e) {
                                // console.warn('Failed to parse SSE event:', data, e);
                            }
                        }
                    }
                }
            } catch (readError: any) {
                if (readError.name === 'AbortError') return; // User cancelled
                throw readError; // Re-throw to trigger retry
            }

        } catch (err: any) {
            if (err.name === 'AbortError') return;

            // Network Error or Disconnect -> Retry if we have thread_id
            if (currentThreadId && retryCount < MAX_RETRIES) {
                retryCount++;
                const delay = 3000 * retryCount; // Backoff
                console.warn(`Stream disconnected. Retrying in ${delay}ms... (Attempt ${retryCount}/${MAX_RETRIES})`);

                callbacks.onLog?.(`Connection interrupted. Reconnecting (Attempt ${retryCount})...`);

                // Wait and Retry
                await new Promise(resolve => setTimeout(resolve, delay));
                if (!controller.signal.aborted) {
                    await connect(); // Recursive logic
                }
            } else {
                // Fatal
                callbacks.onError?.(err.message || "Network Error");
            }
        }
    };

    // Start initial connection
    connect();

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
