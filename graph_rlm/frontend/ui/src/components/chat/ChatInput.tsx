
import React, { useState, useRef, useEffect } from 'react';
import { Send, Square } from 'lucide-react';

interface ChatInputProps {
    onSend: (message: string) => void;
    onStop: () => void;
    isProcessing: boolean;
    placeholder?: string;
    value?: string;
    onChange?: (value: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({ onSend, onStop, isProcessing, placeholder = "Type a message...", value, onChange }) => {
    // Internal state fallback if uncontrolled (backwards compat)
    const [internalInput, setInternalInput] = useState("");

    // Derived values
    const input = value !== undefined ? value : internalInput;
    const setInput = onChange || setInternalInput;

    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const handleSubmit = () => {
        if (!input.trim()) return;
        onSend(input);
        setInput("");
        // Reset height
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    // Auto-resize
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [input]);

    return (
        <div className="p-4 bg-black/50 border-t border-slate-800 backdrop-blur-sm">
            <div className="max-w-4xl mx-auto flex items-end gap-2 bg-slate-900 border border-slate-700 rounded-xl p-2 focus-within:border-blue-500/50 focus-within:shadow-[0_0_15px_rgba(59,130,246,0.1)] transition-all">
                <textarea
                    ref={textareaRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder={placeholder}
                    className="flex-1 bg-transparent border-none outline-none text-slate-200 text-sm p-2 resize-none max-h-[200px] overflow-y-auto placeholder:text-slate-600"
                    rows={1}
                    disabled={isProcessing}
                />

                <div className="pb-1 pr-1">
                    {isProcessing ? (
                        <button
                            onClick={onStop}
                            className="bg-red-500 hover:bg-red-600 text-white p-2 rounded-lg transition-colors flex items-center justify-center shadow-lg shadow-red-500/20"
                            title="Stop Generation"
                        >
                            <Square size={16} fill="currentColor" />
                        </button>
                    ) : (
                        <button
                            onClick={handleSubmit}
                            disabled={!input.trim()}
                            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white p-2 rounded-lg transition-colors flex items-center justify-center shadow-lg shadow-blue-600/20"
                            title="Send Message"
                        >
                            <Send size={16} />
                        </button>
                    )}
                </div>
            </div>
            <div className="text-center mt-2">
                <p className="text-[10px] text-slate-600">Graph RLM v2.0 • Recursive Logic Machine</p>
            </div>
        </div>
    );
};
