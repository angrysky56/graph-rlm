import React, { useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { User, Cpu, Zap, AlertTriangle } from "lucide-react";

interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  thinking?: boolean;
  isChain?: boolean;
  code?: string;
  output?: string;
  tag?: string; // Added for trace coloring
}

const TAG_COLORS: Record<string, string> = {
  AGENT: "text-blue-400",
  LLM: "text-yellow-400",
  REPL: "text-emerald-400",
  SHEAF: "text-fuchsia-400",
  DB: "text-cyan-400",
  RLM: "text-blue-400",
  ERROR: "text-red-400",
  EVENT: "text-slate-400",
};

interface ChatInterfaceProps {
  messages: ChatMessage[];
  isProcessing: boolean;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  isProcessing,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isProcessing]);

  // Flattened Messages (No Grouping)
  const flattenedMessages = messages;

  if (messages.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-500 opacity-50 select-none">
        <Cpu size={48} className="mb-4 text-slate-700" />
        <p className="text-sm font-mono">SYSTEM ONLINE. AWAITING INPUT.</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {flattenedMessages.map((msg, i) => {
        // Determine style based on role/type
        const isUser = msg.role === "user";
        const isCode = !!msg.code;

        if (isCode) {
          return (
            <div
              key={i}
              className="flex flex-col gap-2 mb-4 bg-slate-950/50 p-2 rounded border border-slate-800"
            >
              <div className="flex gap-2 text-blue-400 font-mono text-xs">
                <span className="select-none opacity-50">&gt;&gt;&gt;</span>
                <span className="whitespace-pre-wrap">{msg.code}</span>
              </div>
              <div className="text-emerald-400 font-mono text-xs whitespace-pre-wrap pl-6 border-l-2 border-slate-800 ml-1">
                {msg.output}
              </div>
            </div>
          );
        }

        const isRejected =
          typeof msg.content === "string" &&
          (msg.content.includes(
            "REJECTED: Structural Verification Violation",
          ) ||
            msg.content.startsWith("Return Value: REJECTED"));

        if (isRejected) {
          return (
            <details key={i} className="group mb-4 w-full max-w-4xl mx-auto">
              <summary className="flex items-center gap-2 cursor-pointer select-none rounded-xl px-4 py-2.5 border text-orange-400 border-orange-800/50 bg-orange-950/30 transition-all hover:brightness-125 shadow-lg shadow-black/20">
                <AlertTriangle size={12} className="shrink-0 opacity-70" />
                <span className="text-[10px] font-black uppercase tracking-[0.15em] shrink-0">
                  VALIDATION REJECTED
                </span>
                <span className="text-[11px] opacity-80 truncate ml-2">
                  Attempt failed system guardrails. System will self-correct.
                </span>
              </summary>
              <div className="mt-1 ml-6 px-4 py-3 rounded-lg border text-orange-300 border-orange-800/50 bg-orange-950/30 text-[12px] leading-relaxed font-mono whitespace-pre-wrap">
                {msg.content}
              </div>
            </details>
          );
        }

        if (msg.thinking) {
          const tagColor = msg.tag
            ? TAG_COLORS[msg.tag] || "text-slate-500"
            : "text-slate-500";
          return (
            <div
              key={i}
              className={`text-xs font-mono mb-2 ${tagColor} flex gap-2 ml-10`}
            >
              <span className="select-none opacity-50">
                {msg.tag ? `[${msg.tag}]` : "#"}
              </span>
              <span>{msg.content}</span>
            </div>
          );
        }

        // Standard Message (User or Assistant Final)
        return (
          <div
            key={i}
            className={`flex gap-4 ${isUser ? "flex-row-reverse" : ""}`}
          >
            {/* Avatar */}
            <div
              className={`mt-1 w-8 h-8 rounded shrink-0 flex items-center justify-center border ${
                isUser
                  ? "bg-blue-900/20 border-blue-800 text-blue-400"
                  : "bg-indigo-900/20 border-indigo-800 text-indigo-400"
              }`}
            >
              {isUser ? <User size={16} /> : <Zap size={16} />}
            </div>

            {/* Content Bubble */}
            <div
              className={`flex flex-col max-w-[85%] ${isUser ? "items-end" : "items-start"}`}
            >
              <div
                className={`px-4 py-3 rounded-xl border text-sm shadow-sm ${
                  isUser
                    ? "bg-slate-800 border-slate-700 text-slate-100"
                    : "bg-slate-900 border-slate-800 text-slate-200 font-mono tracking-tight"
                }`}
              >
                <div className="prose prose-invert prose-sm max-w-none leading-relaxed">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
              <span className="text-[10px] text-slate-600 mt-1 px-1">
                {new Date(msg.timestamp).toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
          </div>
        );
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
            <span className="text-xs text-emerald-500/80 font-mono ml-2">
              THINKING
            </span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
};
