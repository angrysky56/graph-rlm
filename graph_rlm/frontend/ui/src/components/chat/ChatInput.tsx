import React, { useState, useRef, useEffect } from "react";
import { Send, Square } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface ChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isProcessing: boolean;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onStop,
  isProcessing,
  placeholder = "Type a message...",
  value,
  onChange,
}) => {
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
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-resize
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [input]);

  return (
    <div className="p-4 bg-transparent border-t border-slate-800/50 backdrop-blur-md sticky bottom-0 z-10 w-full">
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="max-w-4xl mx-auto flex items-end gap-3 bg-slate-900/60 backdrop-blur-xl border border-slate-700/50 rounded-2xl p-2 focus-within:border-indigo-500/50 focus-within:shadow-[0_0_30px_rgba(99,102,241,0.15)] transition-all"
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="flex-1 bg-transparent border-none outline-none text-slate-100 text-[14px] p-3 resize-none max-h-[200px] overflow-y-auto placeholder:text-slate-500 font-medium tracking-wide"
          rows={1}
          disabled={isProcessing}
        />

        <div className="pb-1 pr-1 shrink-0 flex items-center justify-center">
          <AnimatePresence mode="popLayout" initial={false}>
            {isProcessing ? (
              <motion.button
                key="stop-btn"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={onStop}
                className="bg-red-500/20 hover:bg-red-500/30 text-red-400 p-3 rounded-xl transition-colors flex items-center justify-center border border-red-500/30"
                title="Stop Generation"
              >
                <Square size={18} fill="currentColor" />
              </motion.button>
            ) : (
              <motion.button
                key="send-btn"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSubmit}
                disabled={!input.trim()}
                className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-colors flex items-center justify-center shadow-lg shadow-indigo-600/20"
                title="Send Message"
              >
                <Send
                  size={18}
                  className={
                    input.trim() ? "translate-x-[2px] -translate-y-[2px]" : ""
                  }
                />
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-center mt-3"
      >
        <p className="text-[10px] text-slate-500 tracking-[0.2em] font-bold uppercase">
          NEXUS Hybrid Engine • Recursive Logic Machine
        </p>
      </motion.div>
    </div>
  );
};
