
import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Bot, AlertTriangle, CheckCircle, Terminal } from 'lucide-react';

interface ChatEntry {
    type: 'input' | 'output' | 'info' | 'error';
    content: string;
    timestamp: number;
    style?: 'code' | 'thinking' | 'trace' | 'success' | 'error' | 'report';
    isStreaming?: boolean;
    role?: string; // Sometimes used for explicit role
}

interface ChatHistoryProps {
    entries: ChatEntry[];
}

export const ChatHistory: React.FC<ChatHistoryProps> = ({ entries }) => {
    const bottomRef = useRef<HTMLDivElement>(null);

    // Filter out thinking/trace logs for the "Clean" chat view
    // We only want: User Inputs, Final Answers, Error. Code Execution Results Go to the top left sidepanel
    // The user asked for "Final responses show in the middle panel".
    // "Thinking" is usually in the scratchpad or terminal log.
    const displayEntries = entries.filter(e => {
        // Essential Filters
        const isSubstantive = e.style === 'report' || e.style === 'success' || e.style === 'thinking';
        const isUser = e.type === 'input';
        const isStandardOutput = e.type === 'output' && !e.style;
        const isMainError = e.type === 'error' && e.style !== 'trace';

        return isUser || isSubstantive || isMainError || isStandardOutput;
    });

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [displayEntries.length, displayEntries[displayEntries.length-1]?.content]);

    return (
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 custom-scrollbar min-w-0">
            {displayEntries.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-slate-600 opacity-50">
                    <Bot size={48} className="mb-4" />
                    <p>Start a conversation to begin...</p>
                </div>
            )}

            {displayEntries.map((entry, i) => {
                const isUser = entry.type === 'input';
                const isError = entry.type === 'error' || entry.style === 'error';
                const isCode = entry.style === 'code';
                const isSuccess = entry.style === 'success';

                return (
                    // Added min-w-0 to prevent flex item expansion causing overflow
                    <div key={i} className={`flex gap-4 ${isUser ? 'flex-row-reverse' : ''} w-full max-w-4xl mx-auto min-w-0`}>
                        {/* Avatar */}
                        <div className={`mt-1 shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                            isUser ? 'bg-blue-600 text-white' :
                            isError ? 'bg-red-900/50 text-red-500' :
                            isSuccess ? 'bg-emerald-900/50 text-emerald-500' :
                            'bg-slate-800 text-slate-400'
                        }`}>
                            {isUser ? <User size={16} /> :
                             isError ? <AlertTriangle size={16} /> :
                             isSuccess ? <CheckCircle size={16} /> :
                             isCode ? <Terminal size={14} /> :
                             <Bot size={16} />}
                        </div>

                        {/* Message Bubble - Added max-w-full and break-words */}
                        <div className={`flex-1 min-w-0 max-w-full ${isUser ? 'text-right' : ''}`}>
                            <div className={`inline-block text-left rounded-2xl px-6 py-4 text-sm leading-relaxed shadow-sm max-w-full ${
                                isUser ? 'bg-blue-600 text-white rounded-tr-sm' :
                                isError ? 'bg-red-950/30 border border-red-900/50 text-red-200 rounded-tl-sm w-full' :
                                isCode ?  'bg-[#0d1117] border border-slate-800 rounded-tl-sm font-mono text-xs w-full overflow-x-auto' :
                                isSuccess ? 'bg-emerald-950/20 border border-emerald-900/30 text-slate-200 rounded-tl-sm w-full' :
                                'bg-slate-900/80 border border-slate-800 text-slate-200 rounded-tl-sm w-full'
                            }`}>
                                {isCode ? (
                                    <pre className="whitespace-pre-wrap break-words max-w-full overflow-x-auto">{entry.content}</pre>
                                ) : (
                                    <div className={`markdown-content break-words max-w-full ${isUser ? 'text-white' : 'text-slate-300'}`}>
                                       <ReactMarkdown>{entry.content}</ReactMarkdown>
                                    </div>
                                )}
                            </div>
                            <div className={`mt-1 text-[10px] text-slate-600 ${isUser ? 'text-right' : 'text-left'}`}>
                                {new Date(entry.timestamp).toLocaleTimeString()}
                            </div>
                        </div>
                    </div>
                );
            })}
            <div ref={bottomRef} />
        </div>
    );
};
