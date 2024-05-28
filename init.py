import sys
import subprocess
import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

AUDIO_FILE_PATH = None
LANGUAGE = "English"  # Default language


def print_help():
    help_message = """
Usage: python script.py <path_to_audio_file> [options]

Options:
  -h, --help         Show this help message and exit
  -l, --language     Specify the language for the summary (default is English)

Description:
  This script converts an audio file to FLAC format, transcribes the audio using OpenAI's Whisper model,
  and summarizes the transcription using OpenAI's GPT-4 model. The converted FLAC file is deleted after processing.

Arguments:
  <path_to_audio_file>  Path to the audio file you want to transcribe and summarize.
"""
    print(help_message)


# Ensure the audio file path is provided or help is requested
if len(sys.argv) < 2 or "-h" in sys.argv or "--help" in sys.argv:
    print_help()
    sys.exit(0)


for i, arg in enumerate(sys.argv):
    if arg not in ("-l", "--language"):
        if AUDIO_FILE_PATH is None and arg not in (sys.argv[0], "-h", "--help"):
            AUDIO_FILE_PATH = arg
    if arg in ("-l", "--language") and i + 1 < len(sys.argv):
        LANGUAGE = sys.argv[i + 1]

if AUDIO_FILE_PATH is None:
    print("Error: Path to audio file is required.")
    print_help()
    sys.exit(1)


def convert_audio_file(file_path):
    converted_file = "converted_audio.flac"
    subprocess.run(
        ["ffmpeg", "-i", file_path, "-ar", "16000", "-ac", "1", converted_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return converted_file


def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return response.text


def summarize_transcription(transcription_text):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Please summarize this text in {LANGUAGE}"},
            {"role": "user", "content": transcription_text},
        ],
    )
    return response.choices[0].message.content


def main():
    try:
        converted_file_path = convert_audio_file(AUDIO_FILE_PATH)
        transcription = transcribe_audio(converted_file_path)
        summary = summarize_transcription(transcription)

        print("Transcription:")
        print(transcription, "\n")
        print("Summary:")
        print(summary, "\n")
    finally:
        if os.path.exists(converted_file_path):
            os.remove(converted_file_path)


if __name__ == "__main__":
    main()
