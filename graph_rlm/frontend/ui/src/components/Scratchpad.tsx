import React from 'react';

interface ScratchpadProps {
    scratchpadText: string;
}

/**
 * Displays the ACTUAL scratchpad text that the agent sees.
 * This is the verbatim output of build_scratchpad(), not a prettified version.
 */
export const Scratchpad: React.FC<ScratchpadProps> = ({ scratchpadText }) => {
    return (
        <div className="flex flex-col h-full bg-slate-900/80">


            {/* Scratchpad Text - VERBATIM from build_scratchpad() */}
            <div className="flex-1 overflow-y-auto p-4 font-mono text-xs leading-relaxed">
                {scratchpadText ? (
                    <pre className="whitespace-pre-wrap text-slate-300 break-words">
                        {scratchpadText}
                    </pre>
                ) : (
                    <div className="text-slate-600 italic text-center mt-8 text-sm">
                        Scratchpad will appear when the agent starts processing...
                    </div>
                )}
            </div>
        </div>
    );
};
