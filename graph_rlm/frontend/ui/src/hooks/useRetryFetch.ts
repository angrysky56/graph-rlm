import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * A hook that retries a fetch function with exponential backoff on failure.
 * Retries for a maximum period of time.
 */
export function useRetryFetch<T>(
    fetchFn: () => Promise<T>,
    options: {
        maxRetries?: number;
        initialDelay?: number;
        maxDelay?: number;
        onSuccess?: (data: T) => void;
        onError?: (error: any) => void;
    } = {}
) {
    const {
        maxRetries = 10,
        initialDelay = 1000,
        maxDelay = 10000,
        onSuccess,
        onError
    } = options;

    const [data, setData] = useState<T | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<any>(null);
    const [retryCount, setRetryCount] = useState(0);

    const timerRef = useRef<number | null>(null);

    const executeFetch = useCallback(async () => {
        try {
            const result = await fetchFn();
            setData(result);
            setLoading(false);
            setError(null);
            if (onSuccess) onSuccess(result);
        } catch (err) {
            setError(err);
            if (retryCount < maxRetries) {
                const nextDelay = Math.min(initialDelay * Math.pow(2, retryCount), maxDelay);
                console.log(`Fetch failed, retrying in ${nextDelay}ms (attempt ${retryCount + 1}/${maxRetries})`);

                timerRef.current = window.setTimeout(() => {
                    setRetryCount(prev => prev + 1);
                }, nextDelay);
            } else {
                setLoading(false);
                if (onError) onError(err);
            }
        }
    }, [fetchFn, retryCount, maxRetries, initialDelay, maxDelay, onSuccess, onError]);

    useEffect(() => {
        executeFetch();
        return () => {
            if (timerRef.current) window.clearTimeout(timerRef.current);
        };
    }, [executeFetch]);

    return { data, loading, error, retryCount, refresh: () => setRetryCount(0) };
}
