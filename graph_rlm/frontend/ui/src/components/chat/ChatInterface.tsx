
import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { User, Cpu, Zap } from 'lucide-react';

interface ChatMessage {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: number;
    thinking?: boolean;
    isChain?: boolean;
    code?: string;
    output?: string;
}

interface ChatInterfaceProps {
    messages: ChatMessage[];
    isProcessing: boolean;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ messages, isProcessing }) => {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isProcessing]);

    // Group messages
    const groupedMessages: (ChatMessage | { type: 'chain', messages: ChatMessage[] })[] = [];
    let currentChain: ChatMessage[] = [];

    messages.forEach(msg => {
        if (msg.isChain) {
            currentChain.push(msg);
        } else {
            if (currentChain.length > 0) {
                groupedMessages.push({ type: 'chain', messages: [...currentChain] });
                currentChain = [];
            }
            groupedMessages.push(msg);
        }
    });
    if (currentChain.length > 0) {
        groupedMessages.push({ type: 'chain', messages: [...currentChain] });
    }

    if (messages.length === 0) {
        return (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-50 select-none">
                <Cpu size={48} className="mb-4 text-slate-700" />
                <p className="text-sm font-mono">SYSTEM ONLINE. AWAITING INPUT.</p>
            </div>
        );
    }

    return (
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {groupedMessages.map((item, i) => {
                if ('type' in item && item.type === 'chain') {
                    // Render Chain Group
                    const chainGroup = item as { type: 'chain', messages: ChatMessage[] };
                    return (
                        <div key={i} className="flex gap-4">
                            <div className="mt-1 w-8 h-8 rounded shrink-0 flex items-center justify-center border bg-emerald-900/20 border-emerald-800 text-emerald-400">
                                <Zap size={16} />
                            </div>
                            <div className="flex-1 max-w-[90%] flex flex-col">
                                <div className="bg-slate-950 border border-slate-800 rounded-lg overflow-hidden shadow-sm">
                                    <div className="bg-slate-900/50 px-3 py-2 border-b border-slate-800 flex justify-between items-center">
                                        <span className="text-xs font-bold text-emerald-500 tracking-wider flex items-center gap-2">
                                            REASONING CHAIN
                                            <span className="bg-emerald-900/40 text-emerald-300 px-1.5 py-0.5 rounded-full text-[10px]">{chainGroup.messages.length} STEPS</span>
                                        </span>
                                    </div>
                                    <div className="max-h-[500px] overflow-y-auto p-4 custom-scrollbar bg-black font-mono text-xs text-slate-300">
                                        {chainGroup.messages.map((msg, idx) => {
                                            if (msg.code) {
                                                return (
                                                    <div key={idx} className="mb-3">
                                                        <div className="flex gap-2 text-blue-400 mb-1">
                                                            <span className="select-none opacity-50">&gt;&gt;&gt;</span>
                                                            <span className="whitespace-pre-wrap">{msg.code}</span>
                                                        </div>
                                                        <div className="text-slate-400 whitespace-pre-wrap pl-6 border-l-2 border-slate-800 ml-1">
                                                            {msg.output}
                                                        </div>
                                                    </div>
                                                );
                                            } else {
                                                // Thinking / Log
                                                return (
                                                    <div key={idx} className="mb-2 text-slate-500 italic flex gap-2">
                                                        <span className="select-none opacity-50">#</span>
                                                        <span>{msg.content}</span>
                                                    </div>
                                                );
                                            }
                                        })}
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                } else {
                    // Render Standard Message
                    const msg = item as ChatMessage;
                    return (
                        <div key={i} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                            {/* Avatar */}
                            <div className={`mt-1 w-8 h-8 rounded shrink-0 flex items-center justify-center border ${msg.role === 'user'
                                ? 'bg-blue-900/20 border-blue-800 text-blue-400'
                                : 'bg-emerald-900/20 border-emerald-800 text-emerald-400'
                                }`}>
                                {msg.role === 'user' ? <User size={16} /> : <Zap size={16} />}
                            </div>

                            {/* Content Bubble */}
                            <div className={`flex flex-col max-w-[80%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                <div className={`px-4 py-3 rounded-2xl border text-sm shadow-sm ${msg.role === 'user'
                                    ? 'bg-slate-800 border-slate-700 text-slate-100 rounded-tr-none'
                                    : 'bg-slate-900 border-slate-800 text-slate-200 rounded-tl-none font-mono tracking-tight'
                                    }`}>
                                    <div className="prose prose-invert prose-sm max-w-none leading-relaxed">
                                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                                    </div>
                                </div>
                                <span className="text-[10px] text-slate-600 mt-1 px-1">
                                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                </span>
                            </div>
                        </div>
                    );
                }
            })}

            {isProcessing && (
                <div className="flex gap-4 animate-pulse">
                    <div className="mt-1 w-8 h-8 rounded shrink-0 flex items-center justify-center border bg-emerald-900/20 border-emerald-800 text-emerald-400">
                        <Zap size={16} />
                    </div>
                    <div className="px-4 py-3 rounded-2xl rounded-tl-none bg-slate-900 border border-slate-800 flex items-center gap-2">
                        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce" />
                        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-75" />
                        <span className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce delay-150" />
                        <span className="text-xs text-emerald-500/80 font-mono ml-2">THINKING</span>
                    </div>
                </div>
            )}
            <div ref={bottomRef} />
        </div>
    );
};
