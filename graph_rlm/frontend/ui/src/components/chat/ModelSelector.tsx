import React, { useMemo, useState, useEffect } from "react";
import { type Model } from "../../api";
import { ChevronDown, ChevronRight } from "lucide-react";

interface ModelSelectorProps {
  models: Model[];
  currentModel: string;
  onSelect: (modelId: string) => void;
  label?: string;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  currentModel,
  onSelect,
  label,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [openProviders, setOpenProviders] = useState<Record<string, boolean>>(
    {},
  );

  const containerRef = React.useRef<HTMLDivElement>(null);

  const selected = models.find((m) => m.id === currentModel) || models[0];

  // Group models by provider
  const modelsByProvider = useMemo(() => {
    const grouped: Record<string, Model[]> = {};
    models.forEach((model) => {
      const provider = model.id.split("/")[0];
      if (!grouped[provider]) {
        grouped[provider] = [];
      }
      grouped[provider].push(model);
    });
    return grouped;
  }, [models]);

  // Auto-open the provider of the current model
  useEffect(() => {
    if (currentModel) {
      const provider = currentModel.split("/")[0];
      setOpenProviders((prev) => ({ ...prev, [provider]: true }));
    }
  }, [currentModel, isOpen]);

  // Handle click outside to close
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen]);

  const toggleProvider = (provider: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setOpenProviders((prev) => ({ ...prev, [provider]: !prev[provider] }));
  };

  const formatPrice = (price: string) => {
    const num = parseFloat(price);
    if (num === 0) return "Free";
    if (num < 0.001) return `$${(num * 1000000).toFixed(2)}/1M`;
    return `$${num.toFixed(6)}`;
  };

  const handleSelect = (modelId: string) => {
    onSelect(modelId);
    setIsOpen(false);
  };

  return (
    <div className="relative flex flex-col gap-1 w-full" ref={containerRef}>
      {label && (
        <span className="text-[10px] text-slate-500 font-mono tracking-wider ml-1">
          {label}
        </span>
      )}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between px-3 py-2 bg-slate-900 border border-slate-700 hover:border-slate-500 rounded text-xs font-medium text-slate-200 transition-colors w-full"
      >
        <div className="flex flex-col items-start truncate">
          <span className="truncate">{selected?.name || "Select Model"}</span>
          <span className="text-[9px] text-slate-500">{selected?.id}</span>
        </div>
        <ChevronDown size={14} className="shrink-0 text-slate-500" />
      </button>

      {isOpen && (
        <div
          className="absolute top-full left-0 mt-1 w-72 bg-slate-900 border border-slate-700 rounded-md shadow-xl z-50 overflow-hidden"
          style={{ maxHeight: "400px", overflowY: "auto" }}
        >
          {models.length === 0 ? (
            <div className="px-4 py-3 text-xs text-slate-500 text-center">
              No models found.
              <br />
              <span className="opacity-75">Check API Key in Backend.</span>
            </div>
          ) : (
            Object.entries(modelsByProvider).map(
              ([provider, providerModels]) => (
                <div
                  key={provider}
                  className="border-b border-slate-800 last:border-0"
                >
                  <div
                    onClick={(e) => toggleProvider(provider, e)}
                    className="sticky top-0 bg-slate-950 px-3 py-2 text-[10px] font-bold text-slate-400 uppercase tracking-wider cursor-pointer hover:bg-slate-900 flex items-center justify-between transition-colors"
                  >
                    <span>{provider}</span>
                    {openProviders[provider] ? (
                      <ChevronDown size={12} />
                    ) : (
                      <ChevronRight size={12} />
                    )}
                  </div>

                  {openProviders[provider] && (
                    <div className="bg-slate-900">
                      {providerModels.map((model) => (
                        <div
                          key={model.id}
                          onClick={() => handleSelect(model.id)}
                          className={`w-full text-left px-3 py-2 hover:bg-slate-800 text-xs border-b border-slate-800/50 cursor-pointer transition-colors ${model.id === currentModel ? "bg-blue-900/20 border-l-2 border-l-blue-500" : "border-l-2 border-l-transparent"
                            }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-medium text-slate-300 truncate">
                              {model.name.split("/").pop()}
                            </span>
                            {model.supports_tools && (
                              <span className="text-[9px] px-1 rounded bg-slate-800 text-slate-500 border border-slate-700">Tools</span>
                            )}
                          </div>

                          <div className="flex items-center justify-between text-[9px] text-slate-600 mt-1">
                            <div>
                              {Math.round(model.context_length / 1000)}k ctx
                            </div>
                            <div className="font-mono">
                              {formatPrice(model.pricing.prompt)} / {formatPrice(model.pricing.completion)}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ),
            )
          )}
        </div>
      )}
    </div>
  );
};
