
import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { RightSidebar } from './RightSidebar';
import { SettingsModal } from '../settings/SettingsModal';

interface LayoutProps {
    children: React.ReactNode;
    onNewChat?: () => void;
    currentModel: string;
    graphData: { nodes: any[], links: any[] };

    onRefreshConfig?: () => void;
    onInjectContent?: (text: string) => void;
    onSelectSession?: (id: string) => void;
    onOpenExplorer?: () => void;
    usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number; };
    terminalEntries?: any[];
    codeEntries?: any[];
    scratchpadText: string;
}

export const Layout: React.FC<LayoutProps> = ({
    children,
    onNewChat,
    currentModel,
    graphData,
    scratchpadText,

    onRefreshConfig,
    onInjectContent,
    onSelectSession,
    onOpenExplorer,
    usage,
    terminalEntries,
    codeEntries
}) => {
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);

    return (
        <div className="flex h-screen bg-black text-slate-100 font-sans overflow-hidden">
            {/* Sidebar (Left Panel: Graph + History) */}
            <div className="shrink-0 z-20 shadow-xl shadow-black/50">
                <Sidebar
                    onNewChat={onNewChat}
                    currentModel={currentModel}
                    onOpenSettings={() => setIsSettingsOpen(true)}
                    onSelectSession={onSelectSession}
                    onOpenExplorer={onOpenExplorer}
                    usage={usage}
                    terminalEntries={terminalEntries}
                    codeEntries={codeEntries}
                />
            </div>

            {/* Main Content Pane (Chat) */}
            <main className="flex-1 flex flex-col relative z-10 min-w-0 bg-slate-900/50">
                {/* Dot Grid Background for Engineering Feel */}
                <div className="absolute inset-0 bg-[radial-gradient(#1e293b_1px,transparent_1px)] [background-size:20px_20px] opacity-20 pointer-events-none" />
                {children}
            </main>

            {/* Right Sidebar (Graph + Tools) */}
            <div className="shrink-0 z-20 shadow-xl shadow-black/50">
                <RightSidebar
                    graphData={graphData}
                    onInjectContent={onInjectContent || (() => { })}
                    scratchpadText={scratchpadText}
                />
            </div>

            <SettingsModal
                isOpen={isSettingsOpen}
                onClose={() => {
                    setIsSettingsOpen(false);
                    if (onRefreshConfig) onRefreshConfig();
                }}

            />
        </div>
    );
};
