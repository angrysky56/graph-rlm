import React, { useState, useEffect } from "react";
import { Save, RefreshCw, AlertTriangle } from "lucide-react";
import { api } from "../../api";

export const SystemPromptEditor: React.FC = () => {
  const [prompt, setPrompt] = useState("");
  const [originalPrompt, setOriginalPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    setIsLoading(true);
    try {
      const config = await api.getConfig();
      if (config.system_prompt) {
        setPrompt(config.system_prompt);
        setOriginalPrompt(config.system_prompt);
      }
    } catch (err) {
      setError("Failed to load configuration");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    try {
      await api.updateConfig({ system_prompt: prompt });
      setOriginalPrompt(prompt);
    } catch (err) {
      setError("Failed to save configuration");
      console.error(err);
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    setPrompt(originalPrompt);
  };

  return (
    <div className="space-y-4 h-full flex flex-col">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-bold text-slate-200">
          System Context Template
        </h3>
        <div className="flex gap-2">
          <button
            onClick={handleReset}
            disabled={prompt === originalPrompt || isLoading}
            className="p-1.5 text-slate-400 hover:text-white disabled:opacity-30 transition-colors"
            title="Reset to last saved"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      <div className="relative flex-1 min-h-[300px]">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full h-full bg-slate-900/50 border border-slate-700/50 rounded-lg p-4 font-mono text-xs text-slate-300 focus:outline-none focus:ring-1 focus:ring-neon-blue/50 resize-none leading-relaxed"
          spellCheck={false}
        />
        {isLoading && (
          <div className="absolute inset-0 bg-slate-900/50 flex items-center justify-center">
            <div className="w-5 h-5 border-2 border-neon-blue border-t-transparent rounded-full animate-spin" />
          </div>
        )}
      </div>

      <div className="bg-alert-yellow/10 border border-alert-yellow/20 rounded-md p-3">
        <div className="flex gap-2 items-start">
          <AlertTriangle
            size={14}
            className="text-alert-yellow shrink-0 mt-0.5"
          />
          <div className="space-y-1">
            <p className="text-[10px] text-alert-yellow font-bold">
              Dynamic Variables
            </p>
            <p className="text-[10px] text-slate-400 leading-normal">
              Use these keys to inject real-time state: <br />
              <code className="text-neon-pink">{"{manifold}"}</code>,{" "}
              <code className="text-neon-pink">{"{valence}"}</code>,{" "}
              <code className="text-neon-pink">{"{arousal}"}</code>,{" "}
              <code className="text-neon-pink">{"{intrinsic_dim}"}</code>,{" "}
              <code className="text-neon-pink">{"{gate}"}</code>,{" "}
              <code className="text-neon-pink">{"{model_id}"}</code>
            </p>
          </div>
        </div>
      </div>

      <div className="flex justify-end pt-2">
        <button
          onClick={handleSave}
          disabled={isSaving || prompt === originalPrompt}
          className="px-4 py-2 bg-neon-blue/10 hover:bg-neon-blue/20 text-neon-blue border border-neon-blue/50 rounded-lg flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isSaving ? (
            <div className="w-3 h-3 border-2 border-current border-t-transparent rounded-full animate-spin" />
          ) : (
            <Save size={14} />
          )}
          Update System Context
        </button>
      </div>

      {error && <p className="text-xs text-red-400 mt-2">{error}</p>}
    </div>
  );
};
