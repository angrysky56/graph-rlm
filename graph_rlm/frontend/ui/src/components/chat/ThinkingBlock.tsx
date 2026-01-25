import React, { useState } from 'react';
import { Lightbulb, ChevronDown, ChevronRight } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ThinkingBlockProps {
    content: string;
}

export const ThinkingBlock: React.FC<ThinkingBlockProps> = ({ content }) => {
    const [isOpen, setIsOpen] = useState(true);

    if (!content) return null;

    return (
        <div className="my-2 border border-blue-900/40 bg-blue-900/5 rounded-md overflow-hidden">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center gap-2 px-3 py-1.5 bg-blue-900/10 text-blue-400 text-xs font-medium hover:bg-blue-900/20 transition-colors"
                title="Toggle Chain of Thought"
            >
                <Lightbulb size={12} />
                <span className="tracking-wide">THOUGHT PROCESS</span>
                <span className="ml-auto opacity-50">
                    {isOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                </span>
            </button>

            {isOpen && (
                <div className="p-3 text-slate-400 text-sm font-mono leading-relaxed bg-black/20 border-t border-blue-900/20">
                    <ReactMarkdown>{content}</ReactMarkdown>
                </div>
            )}
        </div>
    );
};
