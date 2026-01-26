import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  Send,
  User,
  Bot,
  Paperclip,
  Settings as SettingsIcon,
} from "lucide-react";
import { ThinkingBlock } from "./ThinkingBlock";
import { ToolBlock } from "./ToolBlock";
import { ModelSelector } from "./ModelSelector";
import { SettingsModal } from "../settings/SettingsModal";
import { api } from "../../api";

interface Message {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
  toolCalls?: any[]; // Array of tool call objects
}

interface ChatAreaProps {
  sessionId: string;
}

export const ChatArea: React.FC<ChatAreaProps> = ({ sessionId }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  const [logicModel, setLogicModel] = useState<string>(() => {
    return localStorage.getItem("NEXUS_LOGIC_MODEL") || "";
  });
  const [creativeModel, setCreativeModel] = useState<string>(() => {
    return localStorage.getItem("NEXUS_CREATIVE_MODEL") || "";
  });
  const [showSettings, setShowSettings] = useState(false);
  const [usageStats, setUsageStats] = useState({
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    sessionCost: 0,
  });
  const [models, setModels] = useState<any[]>([]); // Store models for pricing

  const bottomRef = useRef<HTMLDivElement>(null);

  // Fetch models and set default if none selected
  useEffect(() => {
    api
      .fetchModels()
      .then((fetchedModels) => {
        setModels(fetchedModels);
        // If no model selected yet, pick the first one for both
        if (fetchedModels.length > 0) {
          if (!logicModel) setLogicModel(fetchedModels[0].id);
          if (!creativeModel) setCreativeModel(fetchedModels[0].id);
        }
      })
      .catch(console.error);
  }, []);

  // Persist model selection to localStorage
  useEffect(() => {
    if (logicModel) localStorage.setItem("NEXUS_LOGIC_MODEL", logicModel);
    if (creativeModel)
      localStorage.setItem("NEXUS_CREATIVE_MODEL", creativeModel);
  }, [logicModel, creativeModel]);

  // Load History when session changes
  useEffect(() => {
    if (!sessionId) return;

    setMessages([]); // Clear while loading
    api
      .getHistory(sessionId)
      .then((history) => {
        if (history && Array.isArray(history)) {
          // Map Chroma history to UI Message format
          const mapped: Message[] = history.map((item: any) => {
            const meta = item.metadata || {};
            let thinking = meta.thinking || "";
            let toolCalls = [];

            try {
              if (meta.tool_calls) {
                toolCalls =
                  typeof meta.tool_calls === "string"
                    ? JSON.parse(meta.tool_calls)
                    : meta.tool_calls;

                // Map tool_calls to the format ToolBlock expects
                // Result from backend metadata is: {id, name, arguments, result}
                // Formatted for UI: {id, function: {name, arguments}, result, status: 'complete'}
                toolCalls = toolCalls.map((tc: any) => ({
                  id: tc.id,
                  index: tc.index ?? 0,
                  function: {
                    name: tc.name,
                    arguments: tc.arguments,
                  },
                  result: tc.result,
                  status: "complete",
                }));
              }
            } catch (e) {
              console.error("Failed to parse tool_calls metadata", e);
            }

            return {
              role: item.role as "user" | "assistant",
              content: item.content,
              thinking: thinking,
              toolCalls: toolCalls,
            };
          });
          setMessages(mapped);
        }
      })
      .catch(console.error);
  }, [sessionId]);

  const scrollToBottom = () =>
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(scrollToBottom, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsStreaming(true);

    const logicModelInfo = models.find((m) => m.id === logicModel);

    // Metadata Check: Reasoning Support (Prioritize logic model for reasoning setting)
    const supportsReasoning =
      logicModelInfo?.supported_parameters?.some(
        (p: string) => p === "reasoning" || p === "include_reasoning",
      ) || false;

    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", thinking: "", toolCalls: [] },
    ]);

    try {
      api.streamChat(
        {
          message: userMsg.content,
          model: logicModel, // Default/fallback for basic handlers
          logic_model: logicModel,
          creative_model: creativeModel,
          session_id: sessionId,
          include_reasoning: supportsReasoning,
        },
        (event) => {
          if (event.type === "usage") {
            console.log("[ChatArea] Usage packet:", event);
            setUsageStats((prev) => {
              // Estimate cost based on logic price (simplification for MVP display)
              const pricing = logicModelInfo?.pricing || {
                prompt: "0",
                completion: "0",
              };
              const inputCost =
                event.prompt_tokens * parseFloat(pricing.prompt) || 0;
              const outputCost =
                event.completion_tokens * parseFloat(pricing.completion) || 0;

              console.log(
                `[ChatArea] Cost calc: In=${event.prompt_tokens} * ${pricing.prompt} = ${inputCost}, Out=${event.completion_tokens} * ${pricing.completion} = ${outputCost}`,
              );

              return {
                inputTokens: prev.inputTokens + event.prompt_tokens,
                outputTokens: prev.outputTokens + event.completion_tokens,
                reasoningTokens:
                  prev.reasoningTokens + (event.reasoning_tokens || 0),
                sessionCost: prev.sessionCost + inputCost + outputCost,
              };
            });
          }

          setMessages((prev) => {
            const newHistory = [...prev];
            const lastMsg = newHistory[newHistory.length - 1];

            if (event.type === "token") {
              lastMsg.content = (lastMsg.content || "") + (event.content || "");
            } else if (event.type === "thinking") {
              lastMsg.thinking =
                (lastMsg.thinking || "") + (event.content || "");
            } else if (event.type === "tool_call_chunk") {
              // Properly accumulate tool call chunks by index
              const toolCalls = event.tool_calls || [];
              for (const tc of toolCalls) {
                const idx = tc.index ?? 0;
                const existingToolCalls = lastMsg.toolCalls || [];

                // Find existing tool call with this index
                const existingIdx = existingToolCalls.findIndex(
                  (t: any) => t.index === idx,
                );

                if (existingIdx === -1) {
                  // New tool call
                  existingToolCalls.push({
                    index: idx,
                    id: tc.id || "",
                    function: {
                      name: tc.function?.name || "",
                      arguments: tc.function?.arguments || "",
                    },
                  });
                } else {
                  // Accumulate into existing
                  const existing = existingToolCalls[existingIdx];
                  if (tc.id && !existing.id) existing.id = tc.id;
                  if (tc.function?.name && !existing.function.name) {
                    existing.function.name = tc.function.name;
                  }
                  if (tc.function?.arguments) {
                    existing.function.arguments += tc.function.arguments;
                  }
                }
                lastMsg.toolCalls = existingToolCalls;
              }
            } else if (event.type === "tool_executing") {
              // Update tool call status to show it's executing
              const tc = (lastMsg.toolCalls || []).find(
                (t: any) => t.id === event.id,
              );
              if (tc) tc.status = "executing";
            } else if (event.type === "tool_result") {
              // Store the tool result
              const tc = (lastMsg.toolCalls || []).find(
                (t: any) => t.id === event.id,
              );
              if (tc) {
                tc.result = event.result;
                tc.status = "complete";
              }
            } else if (event.type === "done") {
              setIsStreaming(false);
            }

            return newHistory;
          });
        },
      );
    } catch (e) {
      console.error(e);
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="h-14 border-b border-slate-800 flex items-center justify-between px-4 bg-slate-900/50 backdrop-blur-sm relative z-20">
        <div className="flex items-center gap-3">
          <ModelSelector
            models={models}
            currentModel={logicModel}
            onSelect={setLogicModel}
            label="LOGIC MANIFOLD"
          />
          <div className="h-8 w-px bg-slate-800 mx-2" />
          <ModelSelector
            models={models}
            currentModel={creativeModel}
            onSelect={setCreativeModel}
            label="CREATIVE MANIFOLD"
          />

          {/* Usage Stats Display */}
          <div className="hidden md:flex items-center gap-4 text-[10px] text-slate-500 font-mono ml-4 border-l border-slate-700 pl-4">
            <div className="flex flex-col">
              <span className="text-slate-400">IN/OUT</span>
              <span>
                {usageStats.inputTokens} / {usageStats.outputTokens}
              </span>
            </div>
            {usageStats.reasoningTokens > 0 && (
              <div className="flex flex-col text-neon-purple">
                <span className="opacity-70">REASONING</span>
                <span className="text-neon-purple">
                  {usageStats.reasoningTokens}
                </span>
              </div>
            )}
            <div className="flex flex-col text-neon-green">
              <span className="opacity-70">SESSION COST</span>
              <span>${usageStats.sessionCost.toFixed(6)}</span>
            </div>
          </div>
        </div>
        <button
          onClick={() => setShowSettings(true)}
          className="p-2 text-slate-400 hover:text-white rounded-md hover:bg-slate-800 transition-colors"
        >
          <SettingsIcon size={18} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-slate-600 opacity-50">
            <Bot size={48} className="mb-4" />
            <p>NEXUS Online. Ready to think.</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-4 ${msg.role === "user" ? "flex-row-reverse" : ""}`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${msg.role === "user" ? "bg-blue-600" : "bg-purple-600"}`}
            >
              {msg.role === "user" ? <User size={16} /> : <Bot size={16} />}
            </div>

            <div
              className={`flex flex-col max-w-[80%] ${msg.role === "user" ? "items-end" : "items-start"}`}
            >
              <div
                className={`rounded-lg p-4 shadow-sm ${msg.role === "user" ? "bg-blue-600/20 text-blue-100" : "bg-slate-800 text-slate-200"}`}
              >
                {msg.thinking && <ThinkingBlock content={msg.thinking} />}

                {msg.toolCalls
                  ?.filter((tool) => tool.function.name)
                  .map((tool, tIdx) => (
                    <ToolBlock
                      key={tIdx}
                      toolName={tool.function.name}
                      args={tool.function.arguments}
                      result={tool.result}
                    />
                  ))}

                <div className="prose prose-invert prose-sm max-w-none">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 bg-slate-900 border-t border-slate-800">
        <div className="max-w-4xl mx-auto relative flex items-end gap-2 bg-slate-800 p-2 rounded-xl border border-slate-700 focus-within:ring-2 ring-blue-500/50 transition-all shadow-lg">
          <button className="p-2 text-slate-400 hover:text-white transition-colors">
            <Paperclip size={20} />
          </button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) =>
              e.key === "Enter" &&
              !e.shiftKey &&
              (e.preventDefault(), handleSend())
            }
            placeholder="Message NEXUS..."
            className="flex-1 bg-transparent border-none focus:ring-0 text-slate-100 placeholder-slate-500 resize-none max-h-32 min-h-[24px] py-2"
            rows={1}
            style={{ height: "auto", minHeight: "44px" }}
          />

          <button
            onClick={handleSend}
            disabled={!input.trim() || isStreaming}
            className="p-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={18} />
          </button>
        </div>
        <div className="text-center text-[10px] text-slate-600 mt-2">
          NEXUS Hybrid Engine • Bicameral Processing Active
        </div>
      </div>

      {showSettings && (
        <SettingsModal
          onClose={() => {
            setShowSettings(false);
            // Refresh models when settings close
            api.fetchModels().then(setModels).catch(console.error);
          } } isOpen={false}        />
      )}
    </div>
  );
};
