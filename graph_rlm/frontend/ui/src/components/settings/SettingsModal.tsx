
import React, { useState, useEffect } from 'react';
import { api, type Model } from '../../api';
import { X, Save, RefreshCw, Brain } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [config, setConfig] = useState<any>({});
  const [loading, setLoading] = useState(false);
  const [models, setModels] = useState<Model[]>([]);
  const [reembedding, setReembedding] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadSettings();
    }
  }, [isOpen]);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const current = await api.getConfig();
      setConfig(current);
      // Fetch models for the ACTIVE provider from config
      const m = await api.listModels(current.LLM_PROVIDER);
      setModels(m);
    } catch (e) { console.error(e) }
    setLoading(false);
  };

  const handleProviderChange = async (newProvider: string) => {
    setConfig({ ...config, provider: newProvider, LLM_PROVIDER: newProvider });
    // Fetch models for the NEW provider immediately
    try {
      const m = await api.listModels(newProvider);
      setModels(m);
    } catch (e) {
      console.error("Failed to fetch models for new provider", e);
    }
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      await api.updateConfig(config);
      // Reload models
      const m = await api.listModels();
      setModels(m);
      alert("Settings Saved & API Reloaded");
      onClose();
    } catch (e) {
      alert("Failed to save settings");
    }
    setLoading(false);
  };

  const handleReembed = async () => {
    if (
      !confirm(
        'This will refresh the semantic embeddings for ALL memories using the current model. It may take some time. Continue?'
      )
    )
      return;

    setReembedding(true);
    try {
      const res = await api.reembedGraph();
      alert(`Graph Re-embedding Complete! Updated ${res.count} memories.`);
    } catch (e: any) {
      alert(`Re-embedding failed: ${e.message || e}`);
    }
    setReembedding(false);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-700 rounded-lg w-full max-w-lg shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex justify-between items-center">
          <h2 className="text-sm font-bold text-slate-100 tracking-wider">SYSTEM SETTINGS</h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto flex-1">

          {/* Provider Selection */}
          <div className="space-y-2">
            <label className="text-xs uppercase font-bold text-slate-500">LLM Provider</label>
            <select
              className="w-full bg-black/50 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-blue-500 outline-none"
              value={config.provider || "ollama"}
              onChange={(e) => handleProviderChange(e.target.value)}
            >
              <option value="ollama">Ollama (Local)</option>
              <option value="openrouter">OpenRouter (Cloud)</option>
              <option value="openai">OpenAI</option>
              <option value="lmstudio">LM Studio</option>
            </select>
          </div>

          {/* Dynamic Fields based on Provider */}
          {config.provider === 'openrouter' && (
            <div className="space-y-2">
              <label className="text-xs uppercase font-bold text-slate-500">OpenRouter API Key</label>
              <div className="flex gap-2">
                <input
                  type="password"
                  className="flex-1 bg-black/50 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-blue-500 outline-none placeholder:text-slate-600"
                  placeholder={config.OPENROUTER_API_KEY ? "•••••••• (Loaded)" : "Enter key (sk-or-...)"}
                  value={config.OPENROUTER_API_KEY || ""}
                  onChange={(e) => setConfig({ ...config, OPENROUTER_API_KEY: e.target.value })}
                />
              </div>
              <p className="text-[10px] text-slate-600">
                Leave blank if set via <code>OPENROUTER_API_KEY</code> environment variable.
              </p>
            </div>
          )}

          {config.provider === 'openai' && (
            <div className="space-y-2">
              <label className="text-xs uppercase font-bold text-slate-500">OpenAI API Key</label>
              <input
                type="password"
                className="w-full bg-black/50 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-blue-500 outline-none"
                placeholder={config.OPENAI_API_KEY ? "••••••••" : "sk-..."}
                value={config.OPENAI_API_KEY || ""}
                onChange={(e) => setConfig({ ...config, OPENAI_API_KEY: e.target.value })}
              />
            </div>
          )}

          {/* Ollama: No Key Needed */}
          {config.provider === 'ollama' && (
            <div className="p-2 bg-blue-900/10 border border-blue-900/30 rounded text-xs text-blue-300">
              Running locally via Ollama. No API key required.
            </div>
          )}

          {/* Model Selection (Chat) */}
          <div className="space-y-2">
            <label className="text-xs uppercase font-bold text-slate-500">Chat Model</label>

            {/* Rich Model Selector */}
            <div className="border border-slate-700 rounded bg-black/30 max-h-[300px] overflow-y-auto">
              {models.length > 0 ? (
                Object.entries(
                  models
                    .filter(m => m.type !== 'embedding')
                    .filter(m => {
                      // Strict filtering to avoid confusion
                      if (config.provider === 'ollama') return m.provider === 'ollama';
                      if (config.provider === 'openai') return m.provider === 'openai';
                      if (config.provider === 'openrouter') return m.provider !== 'ollama' && m.provider !== 'openai';
                      return true;
                    })
                    .reduce((groups, model) => {
                      const provider = model.provider || model.id.split('/')[0] || 'Unknown';
                      const key = provider.charAt(0).toUpperCase() + provider.slice(1);
                      if (!groups[key]) groups[key] = [];
                      groups[key].push(model);
                      return groups;
                    }, {} as Record<string, Model[]>)
                ).sort((a, b) => a[0].localeCompare(b[0])) // Sort providers
                  .map(([provider, providerModels]) => (
                    <div key={provider} className="border-b border-slate-800 last:border-0">
                      <details className="group" open={providerModels.some(m => m.id === (config[config.provider === 'ollama' ? 'OLLAMA_MODEL' : 'OPENROUTER_MODEL']))}>
                        <summary className="p-2 bg-slate-900/50 hover:bg-slate-800 cursor-pointer text-xs font-bold text-slate-300 flex justify-between items-center select-none">
                          <span>{provider} ({providerModels.length})</span>
                          <span className="text-[10px] text-slate-600 group-open:rotate-180 transition-transform">▼</span>
                        </summary>
                        <div className="p-1 space-y-1 bg-black/20">
                          {providerModels.map(m => {
                            const isSelected = m.id === (config[config.provider === 'ollama' ? 'OLLAMA_MODEL' : 'OPENROUTER_MODEL']);
                            // Parse price if string, handle nulls
                            const pPrompt = typeof m.pricing?.prompt === 'string' ? parseFloat(m.pricing.prompt) : 0;
                            const pCompl = typeof m.pricing?.completion === 'string' ? parseFloat(m.pricing.completion) : 0;
                            const promptPrice = pPrompt * 1000000;
                            const complPrice = pCompl * 1000000;

                            return (
                              <div
                                key={m.id}
                                onClick={() => setConfig({ ...config, [config.provider === 'ollama' ? 'OLLAMA_MODEL' : 'OPENROUTER_MODEL']: m.id })}
                                className={`p-2 rounded cursor-pointer flex justify-between items-center group/item transition-all ${isSelected ? 'bg-blue-600/20 border border-blue-500/50' : 'hover:bg-white/5 border border-transparent'}`}
                              >
                                <div className="flex flex-col">
                                  <span className={`text-xs font-medium ${isSelected ? 'text-blue-300' : 'text-slate-300'}`}>
                                    {m.name} {m.supports_tools ? '🛠️' : ''}
                                  </span>
                                  <span className="text-[10px] text-slate-500">
                                    {Math.round((m.context_length || 0) / 1000)}k ctx
                                  </span>
                                </div>
                                <div className="text-right flex flex-col items-end">
                                  <span className="text-[10px] text-slate-400">
                                    ${promptPrice.toFixed(2)} / ${complPrice.toFixed(2)}
                                  </span>
                                  <span className="text-[9px] text-slate-600">per 1M tokens</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </details>
                    </div>
                  ))
              ) : (
                <div className="p-4 text-xs text-slate-500 italic text-center">
                  No models found for this provider. <br />
                  Click <b>SAVE & RELOAD</b> to fetch models.
                </div>
              )}
            </div>
            <div className="text-[10px] text-slate-500 flex justify-between px-1">
              <span>* Prices per Million Tokens</span>
              <span>🛠️ = Supports Tools</span>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs uppercase font-bold text-slate-500">Embedding Model</label>
            {(() => {
              const embeddingKey = config.provider === 'ollama' ? 'OLLAMA_EMBEDDING_MODEL' : 'OPENROUTER_EMBEDDING_MODEL';

              return models.length > 0 ? (
                <select
                  className="w-full bg-black/50 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-blue-500 outline-none"
                  value={config[embeddingKey] || ""}
                  onChange={(e) => setConfig({ ...config, [embeddingKey]: e.target.value })}
                >
                  <option value="" disabled>Select embedding model...</option>
                  {models
                    .filter(m => m.type === 'embedding')
                    .filter(m => {
                      if (config.provider === 'ollama') return m.provider === 'ollama';
                      if (config.provider === 'openrouter') return m.provider !== 'ollama';
                      return true;
                    })
                    .map(m => (
                      <option key={m.id} value={m.id}>
                        {m.name} ({m.provider || 'Cloud'})
                      </option>
                    ))}
                  {/* Fallback option */}
                  <option value="nomic-embed-text">nomic-embed-text (Manual)</option>
                </select>
              ) : (
                <input
                  type="text"
                  className="w-full bg-black/50 border border-slate-700 rounded p-2 text-sm text-slate-200 focus:border-blue-500 outline-none"
                  placeholder="nomic-embed-text"
                  value={config[embeddingKey] || ""}
                  onChange={(e) => setConfig({ ...config, [embeddingKey]: e.target.value })}
                />
              );
            })()}
            <p className="text-[10px] text-slate-600">Required for RLM memory.</p>

            <div className="pt-2">
              <button
                onClick={handleReembed}
                disabled={reembedding || loading}
                className={`w-full py-2 px-3 rounded border flex items-center justify-center gap-2 text-[10px] uppercase font-bold transition-all ${
                  reembedding
                    ? 'bg-purple-900/20 border-purple-500/50 text-purple-300 cursor-wait'
                    : 'bg-slate-800/50 border-slate-700 hover:border-purple-500 text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
              >
               {reembedding ? <RefreshCw className="animate-spin" size={12} /> : <Brain size={12} />}
               {reembedding ? 'Processing Graph...' : 'Re-embed Memories'}
              </button>
            </div>
          </div>

        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 flex justify-between gap-2 bg-slate-900/50">
          {/* Unload Button (Ollama Only) */}
          {config.provider === 'ollama' && (
            <button
              onClick={async () => {
                setLoading(true);
                try {
                  await api.unloadModel(config.OLLAMA_MODEL);
                  alert("Unload request sent.");
                } catch (e) { console.error(e); }
                setLoading(false);
              }}
              className="text-[10px] text-red-400 hover:text-red-300 flex items-center gap-1 border border-red-900/50 px-2 rounded hover:bg-red-900/20"
              title="Unload from GPU to free VRAM"
            >
              <span className="w-2 h-2 bg-red-500 rounded-full" />
              UNLOAD GPU
            </button>
          )}

          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-xs font-bold text-slate-400 hover:text-white"
            >
              CANCEL
            </button>
            <button
              onClick={handleSave}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-xs font-bold transition-all shadow-lg hover:shadow-blue-500/20 disabled:opacity-50"
            >
              {loading ? <RefreshCw className="animate-spin" size={14} /> : <Save size={14} />}
              SAVE & RELOAD
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
