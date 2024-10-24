import torch
import re
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import gradio as gr

# Load the pre-trained SpeechT5 model and processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Function to spell out abbreviations (like API -> A P I)
def spell_out_abbreviations(text):
    # Regular expression to find words in all uppercase (abbreviations)
    def spell(match):
        return ' '.join(list(match.group(0)))  # Insert spaces between letters
    
    # Substitute abbreviations with their spelled-out versions
    return re.sub(r'\b[A-Z]{2,}\b', spell, text)

def text_to_speech(text):
    # Preprocess the text to handle abbreviations
    processed_text = spell_out_abbreviations(text)
    
    # Preprocess the text input with 'text' argument explicitly specified
    inputs = processor(text=[processed_text], return_tensors="pt")

    # Generate random speaker embeddings for TTS
    speaker_embeddings = torch.randn(1, 512)

    # Generate speech from the text input
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings)

    # Use the HiFi-GAN vocoder to convert the generated speech into a waveform
    waveform = vocoder(speech)

    # Detach the tensor from the computation graph and convert to numpy array
    waveform = waveform.squeeze(0).detach().numpy()

    # Save the waveform as a .wav file
    output_file = 'output.wav'
    sf.write(output_file, waveform, 16000)

    return output_file  # Return the file path to the .wav file

# Gradio interface
iface = gr.Interface(
    fn=text_to_speech,
    inputs=gr.Textbox(label="Enter text for TTS.  note:PRESS SUBMIT BUTTON UNTIL SPEECH IS CLEAR"),  # Use gr.Textbox for user input
    outputs=gr.Audio(type="filepath"),  # Use "filepath" to return the .wav file path
    title="Text-to-Speech Converter",
    description="Enter the Text.*Text to be pronounced seperately has to be given in uppercase eg:API"
)

# Launch the interface
iface.launch(share=True)
