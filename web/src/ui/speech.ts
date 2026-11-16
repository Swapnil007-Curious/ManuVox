// Spoken sentences with speechSynthesis, on-device voices only: a network voice would send the
// text off the device, which the privacy promise rules out.

export interface VoiceOption {
  uri: string;
  name: string;
  lang: string;
}

export class Speech {
  readonly supported = typeof speechSynthesis !== "undefined";
  private listeners: (() => void)[] = [];

  constructor() {
    if (this.supported) speechSynthesis.addEventListener("voiceschanged", () => this.listeners.forEach((listener) => listener()));
  }

  onVoices(listener: () => void): void {
    this.listeners.push(listener);
  }

  /** On-device voices, English first. */
  voices(): VoiceOption[] {
    if (!this.supported) return [];
    return speechSynthesis
      .getVoices()
      .filter((voice) => voice.localService)
      .map((voice) => ({ uri: voice.voiceURI, name: voice.name, lang: voice.lang }))
      .sort((a, b) => Number(b.lang.toLowerCase().startsWith("en")) - Number(a.lang.toLowerCase().startsWith("en")) || a.name.localeCompare(b.name));
  }

  /** Speaks `text`; false when there is no on-device voice. */
  speak(text: string, voiceUri: string | null, rate: number): boolean {
    if (!this.supported || !text) return false;
    const voices = speechSynthesis.getVoices().filter((voice) => voice.localService);
    const voice =
      voices.find((candidate) => candidate.voiceURI === voiceUri) ??
      voices.find((candidate) => candidate.lang.toLowerCase().startsWith("en")) ??
      voices[0];
    if (!voice) return false;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = voice;
    utterance.lang = voice.lang;
    utterance.rate = rate;
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
    return true;
  }

  cancel(): void {
    if (this.supported) speechSynthesis.cancel();
  }
}
