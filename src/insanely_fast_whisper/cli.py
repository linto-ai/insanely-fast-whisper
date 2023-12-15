import torch
import json
from transformers import pipeline
from pyannote.audio import Pipeline
import argparse
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from diarization_pipeline import diarize
from result import build_result
import os.path


parser = argparse.ArgumentParser(description="Automatic Speech Recognition")
parser.add_argument(
    "--file-name",
    required=True,
    type=str,
    help="Path or URL to the audio file to be transcribed.",
)
parser.add_argument(
    "--model-name",
    required=False,
    default="openai/whisper-large-v3",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform ASR. (default: openai/whisper-large-v3)",
)
parser.add_argument(
    "--flash",
    required=False,
    type=bool,
    default=False,
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)
parser.add_argument(
    "--batch-size",
    required=False,
    type=int,
    default=24,
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24)",
)
parser.add_argument(
    "--transcript-path",
    required=False,
    default="output.json",
    type=str,
    help="Path to save the transcription output. (default: output.json)",
)
parser.add_argument(
    "--device-id",
    required=False,
    default="0",
    type=str,
    help='Device ID for your GPU. Just pass the device number when using CUDA, or "cpu". (default: "0")',
)
parser.add_argument(
    "--task",
    required=False,
    default="transcribe",
    type=str,
    choices=["transcribe", "translate"],
    help="Task to perform: transcribe or translate to another language. (default: transcribe)",
)
parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="None",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)
parser.add_argument(
    "--timestamp",
    required=False,
    type=str,
    default="chunk",
    choices=["chunk", "word"],
    help="Whisper supports both chunked as well as word level timestamps. (default: chunk)",
)

parser.add_argument(
    "--diarization_model",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)
parser.add_argument(
    "--number_spk",
    required=False,
    type=int,
    help="Number of speakers.",
)

parser.add_argument(
    "--max_spk",
    required=False,
    type=int,
    help="Number of max speakers.",
)

parser.add_argument(
    "--hf_token",
    required=False,
    default="no_token",
    type=str,
    help="Provide a hf.co/settings/token for Pyannote.audio to diarise the audio clips",
)
parser.add_argument(
    "--diarizationmodel",
    required=False,
    default="pyannote/speaker-diarization-3.1",
    type=str,
    help="Name of the pretrained model/ checkpoint to perform diarization. (default: pyannote/speaker-diarization)",
)

def main():
    args = parser.parse_args()
    ts = "word" if args.timestamp == "word" else True
    pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model_name,
            device="cpu" if args.device_id == "cpu" else f"cuda:{args.device_id}",
            model_kwargs={"use_flash_attention_2": args.flash},
        )

    pipe.model = pipe.model.to_bettertransformer()
    language = None if args.language == "None" else args.language

    with Progress(
        TextColumn("🤗 [progress.description]{task.description}"),
        BarColumn(style="yellow1", pulse_style="white"),
        TimeElapsedColumn(),
    ) as progress:
        progress.add_task("[yellow]Transcribing...", total=None)
        
        whisper_transcription_name = os.path.splitext(os.path.basename(args.file_name))[0] + "_whisper" + ".json"
        check_file = os.path.isfile(whisper_transcription_name)
        
        if check_file is False:
            outputs = pipe(
                args.file_name,
                chunk_length_s=30,
                batch_size=args.batch_size,
                generate_kwargs={"task": args.task, "language": language},
                return_timestamps=ts,
            )
            with open(whisper_transcription_name, "w", encoding="utf8") as fp:
                
                json.dump(outputs, fp, ensure_ascii=False)

        else:
            with open(whisper_transcription_name) as json_file:
                outputs = json.load(json_file)

            

    if args.hf_token != "no_token":
        
        speakers_transcript = diarize(args, outputs)
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result(speakers_transcript, outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed & speaker segmented go check it out over here 👉 {args.transcript_path}"
        )
    else:
        with open(args.transcript_path, "w", encoding="utf8") as fp:
            result = build_result([], outputs)
            json.dump(result, fp, ensure_ascii=False)

        print(
            f"Voila!✨ Your file has been transcribed go check it out over here 👉 {args.transcript_path}"
        )

if __name__ == "__main__":
    main()     