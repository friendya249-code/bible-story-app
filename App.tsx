import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, Type, Modality } from "@google/genai";
import { 
  BookOpen, Wand2, ChevronLeft, ChevronRight, Volume2, VolumeX, 
  Loader2, RefreshCw, Youtube, X, Monitor, Smartphone, Coffee, 
  Heart, Sparkles, KeyRound, AlertCircle
} from 'lucide-react';

// ==========================================
// 1. Types & Constants
// ==========================================

export enum AppState {
  SETUP = 'SETUP',
  GENERATING_STORY = 'GENERATING_STORY',
  GENERATING_ASSETS = 'GENERATING_ASSETS',
  READING = 'READING',
  ERROR = 'ERROR'
}

export interface StoryPage {
  pageNumber: number;
  text: string;
  imagePrompt: string;
  imageUrl?: string;
  audioBuffer?: AudioBuffer;
  isImageLoading: boolean;
  isAudioLoading: boolean;
}

export interface Story {
  title: string;
  theme: string;
  pages: StoryPage[];
}

export interface StoryConfig {
  topic: string;
  ageGroup: string;
  tone: string;
  length: 'short' | 'long';
  language: string;
}

export const DEFAULT_CONFIG: StoryConfig = {
  topic: '',
  ageGroup: '5-7세',
  tone: '따뜻한 교훈이 담긴',
  length: 'short',
  language: 'Korean'
};

// ==========================================
// 2. Services (Audio, Gemini, Video)
// ==========================================

// Initialize AI safely
const apiKey = process.env.API_KEY;
const ai = apiKey ? new GoogleGenAI({ apiKey }) : null;

/** Audio Utils */
function decodeBase64(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number = 24000,
  numChannels: number = 1,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

/** Gemini Service */
const generateStoryText = async (config: StoryConfig): Promise<Story> => {
  if (!ai) throw new Error("API Key missing");
  
  const pageCount = config.length === 'long' ? "12 to 15" : "5 to 6";
  const duration = config.length === 'long' ? "3-4 minute" : "1-2 minute";
  
  const prompt = `
    You are a professional children's book author specializing in Bible stories and fairy tales.
    Write a story (approx ${pageCount} pages, suitable for a ${duration} read/video) based on the following topic: "${config.topic}".
    Target Audience: ${config.ageGroup}.
    Tone: ${config.tone}.
    Language: ${config.language} (Ensure the story text is in ${config.language}).

    For each page, provide:
    1. The story text for that page (keep it concise, engaging, and suitable for reading aloud).
    2. A detailed image generation prompt (in English) describing the scene for that page.
    
    CRITICAL INSTRUCTIONS FOR VISUALS (MUST FOLLOW):
    - Art Style: "3D Disney Pixar animation style, hyper-realistic textures, volumetric lighting, 8k resolution, magical atmosphere, cute and expressive characters".
    - CHARACTER CONSISTENCY (VERY IMPORTANT): 
      * In the image prompt for Page 1, you MUST explicitly define the main character's visual details (e.g., 'a young boy with curly red hair, wearing a blue tunic and leather sandals').
      * In EVERY subsequent page's image prompt, you MUST REPEAT this exact physical description to ensure the character looks exactly the same throughout the book. Do not just say "the boy", say "the young boy with curly red hair and blue tunic...".
    - Scene Consistency: Maintain a consistent color palette (warm, golden, pastel tones) and setting style.

    Return the result strictly as valid JSON.
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: prompt,
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          title: { type: Type.STRING, description: "The title of the story" },
          theme: { type: Type.STRING, description: "The central theme or lesson" },
          pages: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                pageNumber: { type: Type.INTEGER },
                text: { type: Type.STRING, description: "The story text for this page" },
                imagePrompt: { type: Type.STRING, description: "Prompt for generating the illustration in English" }
              },
              required: ["pageNumber", "text", "imagePrompt"]
            }
          }
        },
        required: ["title", "theme", "pages"]
      }
    }
  });

  const jsonText = response.text;
  if (!jsonText) throw new Error("No text returned from Gemini");

  const rawData = JSON.parse(jsonText);

  return {
    ...rawData,
    pages: rawData.pages.map((p: any) => ({
      ...p,
      isImageLoading: true,
      isAudioLoading: true
    }))
  };
};

// Helper for delays
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

const generatePageImageFallback = async (prompt: string): Promise<string> => {
    if (!ai) throw new Error("API Key missing");
    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash-image',
      contents: {
        parts: [{ text: prompt }],
      },
      config: {
        responseModalities: [Modality.IMAGE],
      },
    });
    
    const part = response.candidates?.[0]?.content?.parts?.[0];
    if (part?.inlineData) {
        return `data:image/png;base64,${part.inlineData.data}`;
    }
    throw new Error("Fallback image generation failed");
}

const generatePageImage = async (imagePrompt: string, retries = 2): Promise<string> => {
  if (!ai) throw new Error("API Key missing");
  
  try {
    // Try high quality model first
    const response = await ai.models.generateImages({
      model: 'imagen-4.0-generate-001',
      prompt: imagePrompt,
      config: {
        numberOfImages: 1,
        aspectRatio: '1:1',
        outputMimeType: 'image/jpeg'
      }
    });

    const base64ImageBytes = response.generatedImages?.[0]?.image?.imageBytes;
    if (!base64ImageBytes) throw new Error("No image generated");

    return `data:image/jpeg;base64,${base64ImageBytes}`;
  } catch (error) {
    console.warn("Imagen generation failed, attempting fallback...", error);
    
    // If retry is possible, wait and try the fallback
    if (retries > 0) {
      await delay(2000); // Wait 2 seconds before retry
      try {
        return await generatePageImageFallback(imagePrompt);
      } catch (fallbackError) {
         console.warn("Fallback failed, retrying...", fallbackError);
         return generatePageImage(imagePrompt, retries - 1);
      }
    }
    throw error;
  }
};

const generatePageAudio = async (text: string): Promise<AudioBuffer | undefined> => {
  if (!ai) throw new Error("API Key missing");
  if (!text) return undefined;
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Kore' },
          },
        },
      },
    });

    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (!base64Audio) throw new Error("No audio data returned");

    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    const rawBytes = decodeBase64(base64Audio);
    return await decodeAudioData(rawBytes, audioContext, 24000, 1);
  } catch (error) {
    console.error("TTS Generation failed:", error);
    return undefined;
  }
};

/** Video Exporter */
type VideoFormat = 'landscape' | 'portrait';

function wrapText(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, maxWidth: number, lineHeight: number) {
  const words = text.split('');
  let line = '';
  let testLine = '';
  for (let n = 0; n < words.length; n++) {
    testLine = line + words[n];
    const metrics = ctx.measureText(testLine);
    const testWidth = metrics.width;
    if (testWidth > maxWidth && n > 0) {
      ctx.strokeText(line, x, y);
      ctx.fillText(line, x, y);
      line = words[n];
      y += lineHeight;
    } else {
      line = testLine;
    }
  }
  ctx.strokeText(line, x, y);
  ctx.fillText(line, x, y);
}

const loadImage = (src: string): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
};

const exportStoryToVideo = async (
  story: Story, 
  format: VideoFormat,
  onProgress: (msg: string) => void
): Promise<Blob> => {
  const isLandscape = format === 'landscape';
  
  const canvas = document.createElement('canvas');
  canvas.width = isLandscape ? 1920 : 1080;
  canvas.height = isLandscape ? 1080 : 1920;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error("Could not create canvas context");

  const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 44100 });
  const dest = audioCtx.createMediaStreamDestination();
  
  const canvasStream = canvas.captureStream(30);
  const combinedStream = new MediaStream([
    ...canvasStream.getVideoTracks(),
    ...dest.stream.getAudioTracks()
  ]);

  const mimeTypes = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    'video/mp4'
  ];
  const mimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || '';

  if (!mimeType && !MediaRecorder.isTypeSupported('video/webm')) {
      throw new Error("No supported video mime type found in this browser.");
  }

  const chunks: Blob[] = [];
  const mediaRecorder = new MediaRecorder(combinedStream, {
    mimeType: mimeType,
    videoBitsPerSecond: 8000000
  });

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  mediaRecorder.start();

  ctx.textBaseline = "bottom";
  ctx.lineJoin = "round";
  ctx.lineWidth = 4;

  onProgress("오프닝 만드는 중...");
  
  ctx.fillStyle = "#FFFBF5";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  ctx.strokeStyle = "#F59E0B";
  ctx.lineWidth = 10;
  ctx.strokeRect(40, 40, canvas.width - 80, canvas.height - 80);
  
  ctx.textAlign = "center";
  ctx.font = `bold ${isLandscape ? '80px' : '60px'} 'Gowun Batang', serif`;
  ctx.fillStyle = "#451a03";
  ctx.strokeStyle = "white";
  ctx.lineWidth = 6;
  
  const titleY = canvas.height / 2 - 50;
  wrapText(ctx, story.title, canvas.width / 2, titleY, canvas.width - 200, 100);
  
  ctx.font = `${isLandscape ? '40px' : '30px'} 'Noto Sans KR', sans-serif`;
  ctx.fillStyle = "#78350f";
  ctx.fillText("AI Bible Story Weaver", canvas.width / 2, canvas.height / 2 + 80);

  for(let i=0; i<90; i++) {
     await new Promise(r => setTimeout(r, 33));
     ctx.fillStyle = "rgba(0,0,0,0.01)";
     ctx.fillRect(0,0,1,1); 
  }

  ctx.font = `bold ${isLandscape ? '48px' : '40px'} 'Gowun Batang', serif`;
  ctx.lineWidth = 6;
  
  for (let i = 0; i < story.pages.length; i++) {
    const page = story.pages[i];
    onProgress(`${i + 1} / ${story.pages.length} 페이지 녹화 중...`);

    ctx.fillStyle = "#FFFBF5";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (page.imageUrl) {
      try {
        const img = await loadImage(page.imageUrl);
        
        ctx.save();
        ctx.filter = "blur(30px) opacity(40%)";
        const scale = Math.max(canvas.width / img.width, canvas.height / img.height);
        const x = (canvas.width / 2) - (img.width / 2) * scale;
        const y = (canvas.height / 2) - (img.height / 2) * scale;
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
        ctx.restore();

        let mainImgSize = isLandscape ? 800 : 900;
        let mainImgX = (canvas.width - mainImgSize) / 2;
        let mainImgY = isLandscape ? 80 : 350;
        
        ctx.save();
        ctx.shadowColor = "rgba(0,0,0,0.2)";
        ctx.shadowBlur = 30;
        ctx.shadowOffsetY = 10;
        ctx.fillStyle = "white";
        ctx.fillRect(mainImgX - 20, mainImgY - 20, mainImgSize + 40, mainImgSize + 40);
        ctx.restore();

        ctx.drawImage(img, mainImgX, mainImgY, mainImgSize, mainImgSize);
      } catch (e) {
        console.error("Failed to load image for video", e);
      }
    }

    const textY = isLandscape ? 980 : 1500;
    const maxTextWidth = isLandscape ? 1600 : 900;
    
    ctx.strokeStyle = "white";
    ctx.fillStyle = "#292524";
    
    wrapText(ctx, page.text, canvas.width / 2, textY, maxTextWidth, isLandscape ? 70 : 60);

    let durationMs = 3000; 
    if (page.audioBuffer) {
      const source = audioCtx.createBufferSource();
      source.buffer = page.audioBuffer;
      source.connect(dest);
      source.start();
      durationMs = (page.audioBuffer.duration * 1000) + 800;
    }

    const fps = 30;
    const frames = Math.ceil(durationMs / (1000 / fps));
    
    for (let f = 0; f < frames; f++) {
       await new Promise(r => setTimeout(r, 1000 / fps));
       ctx.fillStyle = "rgba(0,0,0,0.01)";
       ctx.fillRect(0, 0, 1, 1);
    }
  }

  mediaRecorder.stop();
  audioCtx.close();

  return new Promise((resolve) => {
    mediaRecorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType || 'video/webm' });
      resolve(blob);
    };
  });
};

// ==========================================
// 3. Sub-Components
// ==========================================

/** Missing API Key Component */
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
        Vercel 또는 로컬 환경 설정에서 키가 감지되지 않았습니다.
      </p>

      <div className="bg-stone-100 rounded-lg p-4 mb-6 text-sm font-mono text-stone-700 break-all border border-stone-200">
        API_KEY=YOUR_GEMINI_API_KEY
      </div>

      <div className="space-y-4">
        <h3 className="font-bold text-stone-800 flex items-center gap-2">
          <KeyRound className="w-4 h-4" />
          해결 방법
        </h3>
        <ol className="list-decimal list-inside space-y-2 text-stone-600 text-sm">
          <li><strong>aistudio.google.com</strong>에서 API 키를 발급받으세요.</li>
          <li>Vercel 대시보드 &gt; Project Settings &gt; Environment Variables로 이동하세요.</li>
          <li>Key에 <code>API_KEY</code>, Value에 복사한 키를 입력하고 Save하세요.</li>
          <li>Redeploy(재배포) 버튼을 누르면 해결됩니다.</li>
        </ol>
      </div>
      
      <div className="mt-8 pt-6 border-t border-stone-100 text-center">
         <a 
           href="https://aistudio.google.com/app/apikey" 
           target="_blank" 
           rel="noreferrer"
           className="inline-flex items-center gap-2 text-amber-600 font-bold hover:underline"
         >
           API 키 발급받으러 가기 →
         </a>
      </div>
    </div>
  </div>
);

/** ConfigForm Component */
interface ConfigFormProps {
  config: StoryConfig;
  onChange: (newConfig: StoryConfig) => void;
  onSubmit: () => void;
  isLoading: boolean;
}

const ConfigForm: React.FC<ConfigFormProps> = ({ config, onChange, onSubmit, isLoading }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    onChange({ ...config, [name]: value });
  };

  return (
    <div className="w-full max-w-md mx-auto bg-white rounded-2xl shadow-xl p-8 border border-amber-100">
      <div className="text-center mb-8">
        <div className="bg-amber-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
          <BookOpen className="w-8 h-8 text-amber-600" />
        </div>
        <h2 className="text-2xl font-bold text-stone-800">나만의 성경 동화 만들기</h2>
        <p className="text-stone-500 mt-2">아이들을 위한 특별한 이야기를 만들어보세요.</p>
      </div>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-stone-700 mb-2">
            이야기 주제 또는 성경 인물
          </label>
          <input
            type="text"
            name="topic"
            value={config.topic}
            onChange={handleChange}
            placeholder="예: 다윗과 골리앗, 노아의 방주"
            className="w-full px-4 py-3 rounded-lg border border-stone-300 focus:ring-2 focus:ring-amber-500 focus:border-amber-500 outline-none transition-all"
            disabled={isLoading}
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-stone-700 mb-2">
              대상 연령
            </label>
            <select
              name="ageGroup"
              value={config.ageGroup}
              onChange={handleChange}
              className="w-full px-4 py-3 rounded-lg border border-stone-300 focus:ring-2 focus:ring-amber-500 outline-none bg-white"
              disabled={isLoading}
            >
              <option value="3-5세">3-5세 (유아)</option>
              <option value="5-7세">5-7세 (유치부)</option>
              <option value="8-10세">8-10세 (초등 저학년)</option>
              <option value="11-13세">11-13세 (초등 고학년)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-stone-700 mb-2">
              이야기 분위기
            </label>
            <select
              name="tone"
              value={config.tone}
              onChange={handleChange}
              className="w-full px-4 py-3 rounded-lg border border-stone-300 focus:ring-2 focus:ring-amber-500 outline-none bg-white"
              disabled={isLoading}
            >
              <option value="따뜻하고 교훈적인">따뜻한 교훈</option>
              <option value="재미있고 활기찬">재미있고 활기찬</option>
              <option value="차분한 잠자리 동화">잠자리 동화</option>
              <option value="모험심이 가득한">모험 가득</option>
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-stone-700 mb-2">
            이야기 길이 (영상 시간)
          </label>
          <select
            name="length"
            value={config.length}
            onChange={handleChange}
            className="w-full px-4 py-3 rounded-lg border border-stone-300 focus:ring-2 focus:ring-amber-500 outline-none bg-white"
            disabled={isLoading}
          >
            <option value="short">짧게 (5-6 페이지, 약 1-2분)</option>
            <option value="long">길게 (12-15 페이지, 약 3-4분)</option>
          </select>
        </div>

        <button
          onClick={onSubmit}
          disabled={isLoading || !config.topic.trim()}
          className={`w-full py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-2 transition-all transform hover:scale-[1.02] active:scale-[0.98]
            ${isLoading || !config.topic.trim() 
              ? 'bg-stone-200 text-stone-400 cursor-not-allowed' 
              : 'bg-amber-600 text-white hover:bg-amber-700 shadow-lg hover:shadow-amber-200'
            }`}
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              <span>이야기 짓는 중...</span>
            </>
          ) : (
            <>
              <Wand2 className="w-5 h-5" />
              <span>동화 만들기 시작</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

/** DonateButton Component */
const DonateButton: React.FC = () => {
  return (
    <a
      href="https://www.buymeacoffee.com/" 
      target="_blank"
      rel="noopener noreferrer"
      className="fixed bottom-6 right-6 z-50 animate-float group"
      aria-label="Support the developer"
    >
      <div className="bg-[#FFDD00] text-stone-900 px-4 py-3 rounded-full shadow-lg hover:shadow-xl transition-all transform hover:scale-105 flex items-center gap-2 font-bold border-2 border-stone-900 cursor-pointer">
        <div className="bg-white p-1.5 rounded-full">
          <Coffee className="w-5 h-5 text-stone-800" />
        </div>
        <span className="hidden md:inline">개발자에게 커피 한 잔</span>
        <span className="md:hidden">후원</span>
        <Heart className="w-4 h-4 text-red-500 fill-red-500 group-hover:scale-125 transition-transform" />
      </div>
    </a>
  );
};

/** StoryReader Component */
interface StoryReaderProps {
  story: Story;
  onRestart: () => void;
  onRegenerateAudio: (pageIndex: number) => Promise<AudioBuffer | undefined>;
}

const StoryReader: React.FC<StoryReaderProps> = ({ story, onRestart, onRegenerateAudio }) => {
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRegeneratingAudio, setIsRegeneratingAudio] = useState(false);
  
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState("");
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);

  const currentPage = story.pages[currentPageIndex];
  const isLastPage = currentPageIndex === story.pages.length - 1;
  const isFirstPage = currentPageIndex === 0;

  useEffect(() => {
    stopAudio();
    setIsPlaying(false);
  }, [currentPageIndex]);

  const playAudio = async () => {
    // Handle missing audio (e.g. after reload)
    if (!currentPage.audioBuffer) {
      if (isRegeneratingAudio) return;
      setIsRegeneratingAudio(true);
      try {
        const newBuffer = await onRegenerateAudio(currentPageIndex);
        if (newBuffer) {
          playAudioBuffer(newBuffer);
        }
      } finally {
        setIsRegeneratingAudio(false);
      }
      return;
    }
    
    playAudioBuffer(currentPage.audioBuffer);
  };

  const playAudioBuffer = async (buffer: AudioBuffer) => {
    stopAudio();

    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    if (audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }

    const source = audioContextRef.current.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContextRef.current.destination);
    
    source.onended = () => setIsPlaying(false);
    
    audioSourceRef.current = source;
    source.start();
    setIsPlaying(true);
  };

  const stopAudio = () => {
    if (audioSourceRef.current) {
      try {
        audioSourceRef.current.stop();
        audioSourceRef.current.disconnect();
      } catch (e) {}
      audioSourceRef.current = null;
    }
    setIsPlaying(false);
  };

  const toggleAudio = () => {
    if (isPlaying) stopAudio();
    else playAudio();
  };

  const handleNext = () => !isLastPage && setCurrentPageIndex(prev => prev + 1);
  const handlePrev = () => !isFirstPage && setCurrentPageIndex(prev => prev - 1);

  const handleExport = async (format: VideoFormat) => {
    // Check if we have all audio
    const missingAudio = story.pages.some(p => !p.audioBuffer);
    if (missingAudio) {
      const proceed = window.confirm("일부 페이지의 오디오가 로드되지 않았습니다. 오디오 없이 영상을 만드시겠습니까? (취소하면 오디오를 먼저 복구할 수 있습니다)");
      if (!proceed) return;
    }

    setIsExportModalOpen(false);
    setIsExporting(true);
    setExportProgress("비디오 생성 준비 중...");
    stopAudio();

    try {
      const videoBlob = await exportStoryToVideo(story, format, (msg) => {
        setExportProgress(msg);
      });

      const url = URL.createObjectURL(videoBlob);
      const a = document.createElement('a');
      a.href = url;
      const ext = videoBlob.type.includes('mp4') ? 'mp4' : 'webm';
      a.download = `${story.title.replace(/\s+/g, '_')}_${format}.${ext}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      alert("동영상이 저장되었습니다! 유튜브 스튜디오에 이 파일을 업로드하세요.");

    } catch (error) {
      console.error("Export failed", error);
      alert("동영상 변환에 실패했습니다. 브라우저 호환성 문제일 수 있습니다.");
    } finally {
      setIsExporting(false);
      setExportProgress("");
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto relative">
      
      {/* Export Processing Overlay */}
      {isExporting && (
        <div className="absolute inset-0 z-50 bg-white/95 backdrop-blur-md rounded-2xl flex flex-col items-center justify-center text-center p-8 border border-amber-200 shadow-xl">
          <Loader2 className="w-16 h-16 text-amber-600 animate-spin mb-6" />
          <h3 className="text-2xl font-bold text-stone-800 mb-2">동영상 만드는 중</h3>
          <p className="text-stone-600 text-lg mb-4">{exportProgress}</p>
          <div className="w-full max-w-xs bg-stone-100 rounded-full h-2 overflow-hidden">
             <div className="h-full bg-amber-500 animate-pulse w-full origin-left"></div>
          </div>
          <p className="text-sm text-stone-400 mt-6">완료되면 자동으로 파일이 다운로드됩니다.</p>
        </div>
      )}

      {/* Format Selection Modal */}
      {isExportModalOpen && !isExporting && (
        <div className="absolute inset-0 z-40 bg-black/20 backdrop-blur-sm rounded-2xl flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl p-6 w-full max-w-sm animate-in zoom-in-95 duration-200">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-bold text-stone-800">영상 포맷 선택</h3>
              <button onClick={() => setIsExportModalOpen(false)} className="p-1 hover:bg-stone-100 rounded-full text-stone-400">
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="space-y-3">
              <button 
                onClick={() => handleExport('landscape')}
                className="w-full flex items-center gap-4 p-4 rounded-xl border-2 border-stone-100 hover:border-amber-500 hover:bg-amber-50 transition-all group text-left"
              >
                <div className="bg-red-100 p-3 rounded-lg group-hover:bg-red-200 transition-colors">
                  <Monitor className="w-6 h-6 text-red-600" />
                </div>
                <div>
                  <div className="font-bold text-stone-800">YouTube 기본 (16:9)</div>
                  <div className="text-xs text-stone-500">일반적인 TV, PC 감상용</div>
                </div>
              </button>

              <button 
                onClick={() => handleExport('portrait')}
                className="w-full flex items-center gap-4 p-4 rounded-xl border-2 border-stone-100 hover:border-amber-500 hover:bg-amber-50 transition-all group text-left"
              >
                <div className="bg-red-100 p-3 rounded-lg group-hover:bg-red-200 transition-colors">
                  <Smartphone className="w-6 h-6 text-red-600" />
                </div>
                <div>
                  <div className="font-bold text-stone-800">YouTube Shorts (9:16)</div>
                  <div className="text-xs text-stone-500">스마트폰 꽉 찬 화면</div>
                </div>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header Navigation */}
      <div className="flex justify-between items-center mb-6 px-2 flex-wrap gap-3">
        <button 
          onClick={onRestart}
          disabled={isExporting}
          className="text-stone-500 hover:text-amber-600 font-medium flex items-center gap-2 text-sm transition-colors disabled:opacity-50"
        >
          <RefreshCw className="w-4 h-4" /> 처음으로
        </button>
        <div className="text-amber-800 font-bold text-lg md:text-xl truncate text-center order-first md:order-none w-full md:w-auto px-4">
          {story.title}
        </div>
        <button
          onClick={() => setIsExportModalOpen(true)}
          disabled={isExporting}
          className="flex items-center gap-2 bg-gradient-to-r from-red-600 to-red-500 hover:from-red-700 hover:to-red-600 text-white px-5 py-2 rounded-full text-sm font-bold shadow-md hover:shadow-lg transition-all transform hover:-translate-y-0.5 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Youtube className="w-5 h-5" />
          유튜브 영상 저장
        </button>
      </div>

      {/* Book Content Area */}
      <div className="bg-white rounded-2xl shadow-2xl overflow-hidden border border-stone-200 aspect-[4/5] md:aspect-[16/9] flex flex-col md:flex-row relative group">
        
        {/* Image Section */}
        <div className="w-full md:w-1/2 bg-amber-50 relative overflow-hidden flex items-center justify-center p-4 md:p-0">
           {currentPage.isImageLoading ? (
             <div className="flex flex-col items-center justify-center text-amber-700/50">
               <Loader2 className="w-12 h-12 animate-spin mb-2" />
               <span className="text-sm font-medium">그림 그리는 중...</span>
             </div>
           ) : currentPage.imageUrl ? (
             <img 
               src={currentPage.imageUrl} 
               alt={`Illustration for page ${currentPageIndex + 1}`}
               className="w-full h-full object-cover md:absolute inset-0 transition-opacity duration-500 animate-in fade-in zoom-in-105"
             />
           ) : (
             <div className="text-stone-400 text-sm">이미지 없음</div>
           )}
           
           {/* Mobile Audio Toggle */}
           <button 
              onClick={toggleAudio}
              disabled={currentPage.isAudioLoading || !currentPage.audioBuffer}
              className={`md:hidden absolute bottom-4 right-4 p-3 rounded-full shadow-lg backdrop-blur-sm transition-all z-10
                ${isPlaying ? 'bg-amber-500 text-white' : 'bg-white/80 text-stone-700'}
                ${(!currentPage.audioBuffer && !currentPage.isAudioLoading) ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              {currentPage.isAudioLoading ? <Loader2 className="w-5 h-5 animate-spin"/> : isPlaying ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
            </button>
        </div>

        {/* Text Section */}
        <div className="w-full md:w-1/2 p-6 md:p-10 flex flex-col justify-center bg-[#fffbf5]">
          <div className="font-story text-lg md:text-2xl leading-loose text-stone-800 whitespace-pre-wrap">
            {currentPage.text}
          </div>
          
          <div className="hidden md:flex mt-8 justify-end">
            <button 
              onClick={toggleAudio}
              disabled={currentPage.isAudioLoading || !currentPage.audioBuffer}
              className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all
                ${isPlaying 
                  ? 'bg-amber-100 border-amber-300 text-amber-800' 
                  : 'bg-white border-stone-200 text-stone-600 hover:bg-stone-50'}
              `}
            >
               {currentPage.isAudioLoading ? (
                 <Loader2 className="w-4 h-4 animate-spin"/>
               ) : isPlaying ? (
                 <><VolumeX className="w-4 h-4" /> <span>멈춤</span></>
               ) : (
                 <><Volume2 className="w-4 h-4" /> <span>읽어주기</span></>
               )}
            </button>
          </div>
        </div>
        
        {/* Navigation Arrows */}
        <div className="absolute top-1/2 -translate-y-1/2 w-full flex justify-between px-4 pointer-events-none z-20">
          <button 
            onClick={handlePrev}
            disabled={isFirstPage || isExporting}
            className={`pointer-events-auto p-3 rounded-full bg-white/90 shadow-lg backdrop-blur-sm hover:bg-white text-stone-800 disabled:opacity-0 transition-all transform hover:scale-110 hover:shadow-xl
            `}
          >
            <ChevronLeft className="w-6 h-6" />
          </button>
          <button 
            onClick={handleNext}
            disabled={isLastPage || isExporting}
            className={`pointer-events-auto p-3 rounded-full bg-white/90 shadow-lg backdrop-blur-sm hover:bg-white text-stone-800 disabled:opacity-0 transition-all transform hover:scale-110 hover:shadow-xl
            `}
          >
            <ChevronRight className="w-6 h-6" />
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mt-8 flex gap-2 justify-center">
        {story.pages.map((_, idx) => (
          <button
            key={idx}
            onClick={() => !isExporting && setCurrentPageIndex(idx)}
            className={`h-2 rounded-full transition-all duration-300 ${
              idx === currentPageIndex ? 'w-10 bg-amber-500' : 'w-2 bg-stone-200 hover:bg-amber-200'
            }`}
            aria-label={`Go to page ${idx + 1}`}
          />
        ))}
      </div>
    </div>
  );
};

const App: React.FC = () => {
    const [appState, setAppState] = useState<AppState>(AppState.SETUP);
    const [config, setConfig] = useState<StoryConfig>(DEFAULT_CONFIG);
    const [story, setStory] = useState<Story | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleConfigSubmit = async () => {
        setAppState(AppState.GENERATING_STORY);
        try {
            const newStory = await generateStoryText(config);
            setStory(newStory);
            
            setAppState(AppState.GENERATING_ASSETS);
            
            // Start generating assets
            generateStoryAssets(newStory);

        } catch (e: any) {
            console.error(e);
            setAppState(AppState.ERROR);
            setError(e.message || "Failed to generate story");
        }
    };

    const generateStoryAssets = async (initialStory: Story) => {
        const pages = initialStory.pages;
        
        const updatePage = (index: number, updates: Partial<StoryPage>) => {
            setStory(prev => {
                if (!prev) return null;
                const newPages = [...prev.pages];
                newPages[index] = { ...newPages[index], ...updates };
                return { ...prev, pages: newPages };
            });
        };

        // Helper to generate assets for a page
        const processPage = async (index: number, page: StoryPage) => {
            const imgP = generatePageImage(page.imagePrompt)
                .then(url => updatePage(index, { imageUrl: url, isImageLoading: false }))
                .catch(e => {
                    console.error(e);
                    updatePage(index, { isImageLoading: false });
                });
            
            const audioP = generatePageAudio(page.text)
                .then(buf => updatePage(index, { audioBuffer: buf, isAudioLoading: false }))
                .catch(e => {
                    console.error(e);
                    updatePage(index, { isAudioLoading: false });
                });
            
            return Promise.all([imgP, audioP]);
        };

        // Wait for first page assets to be ready before showing reader
        await processPage(0, pages[0]);
        
        // Transition to reading immediately after first page
        setAppState(AppState.READING);
        
        // Process remaining pages sequentially to avoid rate limits
        for (let i = 1; i < pages.length; i++) {
            try {
                await processPage(i, pages[i]); 
            } catch (e) {
                console.error(`Failed to generate assets for page ${i}`, e);
            }
            // Add a generous delay to prevent 429 Too Many Requests
            await delay(2000);
        }
    };

    const handleRestart = () => {
        setAppState(AppState.SETUP);
        setStory(null);
        setConfig(DEFAULT_CONFIG);
        setError(null);
    };

    const handleRegenerateAudio = async (pageIndex: number) => {
         if (!story) return undefined;
         const page = story.pages[pageIndex];
         try {
             const buffer = await generatePageAudio(page.text);
             if (buffer) {
                 setStory(prev => {
                     if (!prev) return null;
                     const newPages = [...prev.pages];
                     newPages[pageIndex] = { ...newPages[pageIndex], audioBuffer: buffer, isAudioLoading: false };
                     return { ...prev, pages: newPages };
                 });
             }
             return buffer;
         } catch (e) {
             return undefined;
         }
    };

    return (
    <div className="min-h-screen bg-[#fffbf5] font-sans text-stone-800 selection:bg-amber-200">
      <header className="p-6 border-b border-stone-100 flex justify-between items-center bg-white/50 backdrop-blur-sm sticky top-0 z-30">
        <div className="flex items-center gap-2 text-amber-600">
          <BookOpen className="w-6 h-6" />
          <h1 className="font-bold text-xl tracking-tight">Bible Story Weaver</h1>
        </div>
        <div className="text-xs text-stone-400 font-medium px-3 py-1 bg-stone-100 rounded-full">
          Powered by Google Gemini
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 md:py-12">
        {appState === AppState.SETUP && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
             <ConfigForm 
               config={config} 
               onChange={setConfig} 
               onSubmit={handleConfigSubmit} 
               isLoading={false} 
             />
          </div>
        )}

        {appState === AppState.GENERATING_STORY && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] animate-in fade-in duration-500">
            <div className="relative mb-8">
              <div className="absolute inset-0 bg-amber-200 rounded-full animate-ping opacity-25"></div>
              <div className="relative bg-white p-6 rounded-full shadow-xl border-2 border-amber-100">
                 <Sparkles className="w-12 h-12 text-amber-500 animate-pulse" />
              </div>
            </div>
            <h2 className="text-2xl font-bold text-stone-800 mb-2">이야기를 짓고 있어요</h2>
            <p className="text-stone-500 text-center max-w-md">
              AI 작가가 재미있는 성경 이야기를 생각하고 있습니다.<br/>
              잠시만 기다려주세요. (약 10-20초 소요)
            </p>
          </div>
        )}

        {(appState === AppState.GENERATING_ASSETS) && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] animate-in fade-in duration-500">
            <div className="relative mb-8">
              <div className="absolute inset-0 bg-amber-200 rounded-full blur-xl opacity-50 animate-pulse"></div>
              <div className="relative bg-white p-6 rounded-full shadow-xl">
                <Sparkles className="w-12 h-12 text-amber-500 animate-spin-slow" />
              </div>
            </div>
            <h2 className="text-2xl font-bold text-stone-800 mb-2 font-story">
              삽화를 그리고 목소리를 담고 있습니다...
            </h2>
            <p className="text-stone-500 max-w-xs mx-auto text-center">
              AI가 아이들을 위한 특별한 성경 동화를 만들고 있어요. 잠시만 기다려주세요.
            </p>
          </div>
        )}
        
        {appState === AppState.READING && story && (
           <div className="animate-in zoom-in-95 duration-500">
             <StoryReader 
                story={story} 
                onRestart={handleRestart}
                onRegenerateAudio={handleRegenerateAudio}
             />
           </div>
        )}

        {appState === AppState.ERROR && (
          <div className="text-center py-20 animate-in shake">
            <div className="w-20 h-20 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-6">
              <AlertCircle className="w-10 h-10 text-red-500" />
            </div>
            <h2 className="text-2xl font-bold text-stone-800 mb-2">오류가 발생했습니다</h2>
            <p className="text-stone-500 mb-8">{error || "알 수 없는 오류가 발생했습니다."}</p>
            <button 
              onClick={handleRestart}
              className="px-8 py-3 bg-stone-800 text-white rounded-xl font-bold hover:bg-stone-900 transition-all"
            >
              다시 시작하기
            </button>
          </div>
        )}
      </main>
      
      {!apiKey && <MissingApiKey />}
      <DonateButton />
    </div>
  );
};

export default App;
