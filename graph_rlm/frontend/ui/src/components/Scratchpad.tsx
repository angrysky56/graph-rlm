import React from 'react';
import ReactMarkdown from 'react-markdown';

interface ScratchpadProps {
    scratchpadText: string;
}

/**
 * Displays the ACTUAL scratchpad text that the agent sees.
 * Uses ReactMarkdown to parse the markdown string built by scratchpad_builder.py.
 */
export const Scratchpad: React.FC<ScratchpadProps> = ({ scratchpadText }) => {
    return (
        <div className="flex flex-col h-full bg-[#0d1117]">
            <div className="flex-1 overflow-y-auto p-5 custom-scrollbar">
                {scratchpadText ? (
                    <div className="prose prose-invert prose-sm max-w-none prose-headings:text-indigo-300 prose-headings:font-semibold prose-h1:text-[15px] prose-h2:text-[13px] prose-h3:text-[12px] prose-p:text-[12px] prose-p:leading-relaxed prose-li:text-[12px] prose-strong:text-indigo-200 text-slate-300">
                        <ReactMarkdown
                            components={{
                                code({node, inline, className, children, ...props}: any) {
                                    const match = /language-(\w+)/.exec(className || '');
                                    const isCodeBlock = !inline && match;

                                    if (isCodeBlock) {
                                        return (
                                            <div className="mt-3 mb-3 border border-slate-700/50 rounded-md overflow-hidden bg-[#0d1117]">
                                                <div className="font-bold select-none text-[10px] uppercase tracking-wider text-slate-400 bg-slate-800/40 px-3 py-1.5 border-b border-slate-700/50">
                                                    {match[1]} output
                                                </div>
                                                <div className="p-3 overflow-x-auto text-[11px] leading-relaxed font-mono text-slate-300">
                                                    <code className={className} {...props}>
                                                        {children}
                                                    </code>
                                                </div>
                                            </div>
                                        );
                                    }

                                    return (
                                        <code className="bg-slate-800/60 text-indigo-300 border border-slate-700/30 px-1.5 py-0.5 rounded font-mono text-[11px]" {...props}>
                                            {children}
                                        </code>
                                    );
                                }
                            }}
                        >
                            {scratchpadText}
                        </ReactMarkdown>
                    </div>
                ) : (
                    <div className="text-slate-600 italic text-center mt-8 text-sm">
                        Scratchpad will appear when the agent starts processing...
                    </div>
                )}
            </div>
        </div>
    );
};
