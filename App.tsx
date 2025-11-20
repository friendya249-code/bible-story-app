import React, { useState, useEffect, useRef } from 'react';
import { AppState, Story, StoryConfig, DEFAULT_CONFIG, StoryPage } from './types';
import { generateStoryText, generatePageImage, generatePageAudio } from './services/geminiService';
import { ConfigForm } from './components/ConfigForm';
import { StoryReader } from './components/StoryReader';
import { AlertCircle, KeyRound, Coffee, Heart } from 'lucide-react';
import { DonateButton } from './components/DonateButton';

// --- Missing API Key Component ---
const MissingApiKey: React.FC = () => (
  <div className="min-h-screen flex items-center justify-center bg-stone-900 p-4">
    <div className="bg-white max-w-lg w-full rounded-2xl p-8 shadow-2xl border-l-8 border-amber-500">
      <div className="flex items-center gap-3 mb-6">
        <div className="bg-red-100 p-3 rounded-full">
          <AlertCircle className="w-8 h-8 text-red-600" />
        </div>
        <h1 className="text-2xl font-bold text-stone-800">API 키가 필요해요</h1>
      </div>
      
      <p className="text-stone-600 mb-6 leading-relaxed">
        이 앱을 실행하려면 <strong>Google Gemini API 키</strong>가 필요합니다.<br/>
        <code>.env</code> 파일에 API 키를 설정했는지 확인해주세요.
      </p>

      <div className="bg-stone-100 rounded-lg p-4 mb-6 text-sm font-mono text-stone-700 break-all border border-stone-200">
        API_KEY=YOUR_GEMINI_API_KEY
      </div>
      
      <div className="mt-8 pt-6 border-t border-stone-100 text-center">
         <a 
           href="https://aistudio.google.com/app/apikey" 
           target="_blank" 
           rel="noreferrer"
           className="inline-flex items-center gap-2 text-amber-600 hover:text-amber-800 font-bold"
         >
           <KeyRound className="w-4 h-4" />
           API 키 발급받기
         </a>
      </div>
    </div>
  </div>
);

// --- Main App Component ---
const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>(AppState.SETUP);
  const [config, setConfig] = useState<StoryConfig>(DEFAULT_CONFIG);
  const [story, setStory] = useState<Story | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isGeneratingAssets, setIsGeneratingAssets] = useState(false);
  
  // Check for API Key availability
  const hasApiKey = !!process.env.API_KEY;

  const handleConfigSubmit = async () => {
    setIsLoading(true);
    setAppState(AppState.GENERATING_STORY);

    try {
      // 1. Generate Text Structure
      const newStory = await generateStoryText(config);
      setStory(newStory);
      setAppState(AppState.READING);
      
      // 2. Start Asset Generation (Images/Audio) in background
      // We use a separate function to handle this asynchronously
      generateAssetsSequentially(newStory);
      
    } catch (error) {
      console.error("Failed to generate story:", error);
      alert("이야기를 짓는데 실패했어요. 잠시 후 다시 시도해주세요.");
      setAppState(AppState.SETUP);
    } finally {
      setIsLoading(false);
    }
  };

  const generateAssetsSequentially = async (currentStory: Story) => {
    if (isGeneratingAssets) return;
    setIsGeneratingAssets(true);

    // Create a mutable copy to update state progressively
    let updatedPages = [...currentStory.pages];

    for (let i = 0; i < updatedPages.length; i++) {
      const page = updatedPages[i];
      
      // Skip if already generated
      if (page.imageUrl && page.audioBuffer) continue;

      try {
        // Generate Image if missing
        if (!page.imageUrl) {
          const imageUrl = await generatePageImage(page.imagePrompt);
          
          // Update state immediately for this page
          setStory(prev => {
            if (!prev) return null;
            const newPages = [...prev.pages];
            newPages[i] = { ...newPages[i], imageUrl, isImageLoading: false };
            return { ...prev, pages: newPages };
          });
        }

        // Generate Audio if missing
        if (!page.audioBuffer) {
           const audioBuffer = await generatePageAudio(page.text);
           
           // Update state immediately
           setStory(prev => {
            if (!prev) return null;
            const newPages = [...prev.pages];
            newPages[i] = { ...newPages[i], audioBuffer, isAudioLoading: false };
            return { ...prev, pages: newPages };
          });
        }

        // Small delay to prevent rate limiting
        await new Promise(r => setTimeout(r, 1000));

      } catch (e) {
        console.error(`Asset generation failed for page ${i + 1}`, e);
        // Mark loading as false so we don't show spinner forever
        setStory(prev => {
            if (!prev) return null;
            const newPages = [...prev.pages];
            newPages[i] = { ...newPages[i], isImageLoading: false, isAudioLoading: false };
            return { ...prev, pages: newPages };
        });
      }
    }
    setIsGeneratingAssets(false);
  };

  const handleRestart = () => {
    if (confirm("처음으로 돌아가시겠습니까? 현재 이야기는 사라집니다.")) {
      setStory(null);
      setAppState(AppState.SETUP);
      setConfig(DEFAULT_CONFIG);
    }
  };

  if (!hasApiKey) {
    return <MissingApiKey />;
  }

  return (
    <div className="min-h-screen bg-[#fffbf5] relative">
      <div className="max-w-6xl mx-auto px-4 py-8 min-h-screen flex flex-col">
        
        {/* Header */}
        <header className="mb-8 text-center">
           {/* Only show large header in Setup mode */}
           {appState === AppState.SETUP && (
             <h1 className="text-4xl md:text-5xl font-bold text-amber-800 font-story mb-2">
               바이블 스토리 위버
             </h1>
           )}
        </header>

        {/* Content */}
        <main className="flex-1 flex flex-col items-center justify-center">
          {appState === AppState.SETUP || appState === AppState.GENERATING_STORY ? (
            <ConfigForm 
              config={config} 
              onChange={setConfig} 
              onSubmit={handleConfigSubmit}
              isLoading={isLoading}
            />
          ) : (
            story && <StoryReader story={story} onRestart={handleRestart} />
          )}
        </main>

        <footer className="mt-12 text-center text-stone-400 text-sm pb-4">
          Powered by Google Gemini & Imagen
        </footer>
      </div>
      
      <DonateButton />
    </div>
  );
};

export default App;
