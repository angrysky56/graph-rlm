
import { useEffect, useState, type RefObject } from 'react';

export const useResizeObserver = (ref: RefObject<HTMLElement>) => {
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

    useEffect(() => {
        const target = ref.current;
        if (!target) return;

        const resizeObserver = new ResizeObserver((entries) => {
            if (!entries || !entries.length) return;
            const { width, height } = entries[0].contentRect;

            // Use requestAnimationFrame to avoid loop limit errors
            window.requestAnimationFrame(() => {
                setDimensions({ width, height });
            });
        });

        resizeObserver.observe(target);

        return () => {
            resizeObserver.disconnect();
        };
    }, [ref.current]);

    return dimensions;
};
