
import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { RightSidebar } from './RightSidebar';
import { SettingsModal } from '../settings/SettingsModal';

interface LayoutProps {
    children: React.ReactNode;
    onNewChat?: () => void;
    currentModel: string;
    onSelectModel: (model: any) => void; // Using any or import Model type
    graphData: { nodes: any[], links: any[] };

    onRefreshConfig?: () => void;
    onInjectContent?: (text: string) => void;
}

export const Layout: React.FC<LayoutProps> = ({
    children,
    onNewChat,
    onSelectModel,
    currentModel,
    graphData,

    onRefreshConfig,
    onInjectContent
}) => {
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);

    return (
        <div className="flex h-screen bg-black text-slate-100 font-sans overflow-hidden">
            {/* Sidebar (Left Panel: Graph + History) */}
            <div className="shrink-0 z-20 shadow-xl shadow-black/50">
                <Sidebar
                    onNewChat={onNewChat}
                    currentModel={currentModel}
                    onSelectModel={onSelectModel}
                    onOpenSettings={() => setIsSettingsOpen(true)}
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
