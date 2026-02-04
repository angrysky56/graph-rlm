import React, { useState, useEffect } from "react";
import { api } from "../../api";
import { FileText, Loader, RefreshCw, Trash2 } from "lucide-react";

interface SessionListProps {
  onSelectSession: (id: string) => void;
  className?: string;
  activeSessionId?: string | null;
}

export const SessionList: React.FC<SessionListProps> = ({
  onSelectSession,
  className = "",
  activeSessionId,
}) => {
  const [sessions, setSessions] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchSessions = async () => {
    setLoading(true);
    try {
      const data = await api.getSessions();
      if (Array.isArray(data)) {
        // Sort by created_at desc (which is now last_active from backend)
        const sorted = data.sort((a, b) => {
          return (
            new Date(b.created_at || 0).getTime() -
            new Date(a.created_at || 0).getTime()
          );
        });
        setSessions(sorted);
      }
    } catch (e) {
      console.error("Failed to load sessions", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex justify-between items-center mb-2 px-1">
        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
          Recent Sessions
        </span>
        <button
          onClick={fetchSessions}
          className="text-slate-600 hover:text-emerald-400 transition-colors"
          title="Refresh List"
        >
          <RefreshCw size={12} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto pr-1 space-y-1 custom-scrollbar">
        {loading && (
          <div className="flex justify-center p-4">
            <Loader size={16} className="animate-spin text-slate-600" />
          </div>
        )}

        {!loading && sessions.length === 0 && (
          <div className="text-center text-slate-600 text-xs py-4 italic">
            No sessions found. Start dreaming.
          </div>
        )}

        {sessions.map((s) => {
          const isActive = activeSessionId === s.id;
          return (
            <div
              key={s.id}
              onClick={() => onSelectSession(s.id)}
              className={`
                group p-3 rounded-lg cursor-pointer border transition-all duration-200
                ${
                  isActive
                    ? "bg-blue-900/20 border-blue-500/50 shadow-blue-900/20 shadow-md"
                    : "bg-slate-900/50 border-slate-800 hover:bg-slate-800 hover:border-slate-700"
                }
              `}
            >
              <div className="flex items-start gap-3">
                <div
                  className={`mt-1 p-1 rounded ${
                    isActive
                      ? "bg-blue-500/20 text-blue-400"
                      : "bg-slate-800 text-slate-500 group-hover:text-slate-300"
                  }`}
                >
                  <FileText size={14} />
                </div>
                <div className="flex-1 min-w-0">
                  <h4
                    className={`text-xs font-medium truncate mb-1 ${
                      isActive
                        ? "text-blue-200"
                        : "text-slate-400 group-hover:text-slate-200"
                    }`}
                  >
                    {s.title || "Untitled Session"}
                  </h4>
                  <div className="flex justify-between items-center text-[10px] text-slate-600">
                    <span>
                      {new Date(s.created_at || Date.now()).toLocaleDateString()}
                    </span>
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            if (confirm("Delete this session?")) {
                                api.deleteSession(s.id).then(() => fetchSessions());
                            }
                        }}
                        className="text-slate-700 hover:text-red-400 p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Delete Session"
                    >
                        <Trash2 size={10} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
