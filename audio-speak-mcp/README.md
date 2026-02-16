# audio-speak-mcp

MCP server for text-to-speech output. Gives Claude a voice.

## Overview

This MCP server provides speech synthesis capabilities for AI assistants. It supports multiple TTS engines with a unified interface, allowing flexible selection based on quality and latency requirements.

## Engines

| Engine | Quality | Latency | Requirements |
|--------|---------|---------|--------------|
| **macOS** (`macos`) | Low | Instant | None (built-in `say` command) |
| **Kokoro** (`kokoro`) | Medium | ~2-3s for typical utterance | Apple Silicon, mlx-audio |
| **ElevenLabs** (`elevenlabs`) | High | ~10-30s (API round-trip) | API key, internet connection |

## Installation

```bash
cd audio-speak-mcp

# macOS engine (no extra dependencies)
uv sync

# Kokoro engine (local, Apple Silicon)
uv sync --extra kokoro
python -m unidic download   # Japanese dictionary for MeCab

# ElevenLabs engine (cloud API)
uv sync --extra elevenlabs
```

All engines require `mpv` for audio playback (except macOS which uses the built-in `say` command):

```bash
brew install mpv
```

## Configuration

Set environment variables or create a `.env` file:

### Common

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENGINE` | `macos` | Engine to use: `macos`, `kokoro`, or `elevenlabs` |

### macOS engine

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_VOICE` | `Kyoko` | macOS voice name |
| `TTS_RATE` | (system default) | Speaking rate (words per minute) |

### Kokoro engine

| Variable | Default | Description |
|----------|---------|-------------|
| `KOKORO_VOICE` | `jf_alpha` | Voice preset (see voices below) |
| `KOKORO_MODEL_ID` | `mlx-community/Kokoro-82M-bf16` | Hugging Face model ID |
| `KOKORO_SPEED` | `1.0` | Speaking speed multiplier |
| `KOKORO_LANG_CODE` | `j` | Language: `j` (Japanese), `a` (American English), `b` (British English) |

#### Kokoro voices

Japanese:
- `jf_alpha` — Female (recommended, most training data)
- `jf_gongitsune`, `jf_tebukuro`, `jf_nezumi` — Female
- `jm_kumo` — Male

English:
- `af_heart` — Female (best English quality)
- `af_alloy`, `am_adam` — American English
- `bf_emma`, `bm_george` — British English

### ElevenLabs engine

| Variable | Default | Description |
|----------|---------|-------------|
| `ELEVENLABS_API_KEY` | (required) | ElevenLabs API key |
| `ELEVENLABS_VOICE_ID` | (required) | Default voice ID |
| `ELEVENLABS_MODEL_ID` | `eleven_v3` | ElevenLabs model |

## Tools

### say

Speak text aloud.

```json
{
  "text": "こんにちは",
  "voice": "jf_alpha",
  "rate": 150
}
```

- `text` (required): Text to speak
- `voice` (optional): Voice name/ID (defaults to engine default)
- `rate` (optional): Speaking rate in wpm (macOS engine only)

### get_voices

List available voices for the current engine.

## Claude Code Integration

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "audio-speak": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/audio-speak-mcp",
        "--extra", "kokoro",
        "audio-speak-mcp"
      ],
      "env": {
        "TTS_ENGINE": "kokoro"
      }
    }
  }
}
```

## License

MIT
