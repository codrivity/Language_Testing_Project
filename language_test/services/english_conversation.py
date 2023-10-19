# Import necessary libraries
from pyannote.audio import Pipeline
import librosa
import torch
import re
from textblob import TextBlob
import statistics
from resemblyzer import VoiceEncoder
from spectralcluster import SpectralClusterer
import numpy as np
from spectralcluster import RefinementOptions
from resemblyzer import sampling_rate
import statistics
from pydub import AudioSegment




class EnglishConversationalSpeechtoText():
    def __init__(self, file_name,speaker_diarization_pipeline,summary_pipeline,speech_to_text_pipeline,speech_emotion_model,speech_emotion_feature_extractor,identify_speaker_pipeline):
        # Constructor to initialize the class

        # Initialize the speaker diarization pipeline
        self.speaker_diarization_pipeline = speaker_diarization_pipeline

        # Convert the audio to WAV format to avoid issues with diarization
        temp_audio = AudioSegment.from_wav(file_name)
        temp_audio = temp_audio[1000:]
        temp_audio.export(file_name, format='wav')

        # Apply diarization to the audio file and get the diarization result
        self.diarization = self.speaker_diarization_pipeline(file_name)

        # Load the audio and its sampling rate
        self.audio, self.sr = librosa.load(file_name)

        # Load the emotion audio and its sampling rate (set to 16000)
        self.emotion_audio, self.emotion_sr = librosa.load(file_name, sr=16000)

        # Initialize the voice encoder for speaker clustering
        self.encoder = VoiceEncoder("cpu")

        # Define options for spectral clustering refinement
        refinement_options = RefinementOptions(
            gaussian_blur_sigma=1,
            p_percentile=0.90)

        # Initialize the spectral clusterer for speaker clustering
        self.clusterer = SpectralClusterer(
            min_clusters=3,
            max_clusters=100,
            refinement_options=refinement_options)

        # Get the speech-to-text pipeline
        self.speech_to_text_pipeline = speech_to_text_pipeline

        # Initialize the tokenizer and model for summarization
        self.summary_pipeline = summary_pipeline

        # Load the speech emotion classification model and feature extractor
        self.speech_emotion_model = speech_emotion_model
        self.speech_emotion_feature_extractor = speech_emotion_feature_extractor

        # Load identify speaker pipeline
        self.identify_speaker_pipeline = identify_speaker_pipeline

        # List of emotions
        self.lstEmotions = ['Neutral', 'Happy', 'Angry', 'Sad']


   
    
    def get_response_time(self,final_lst_chat,main_speaker):
        # Function to calculate the response time between speakers
        response_time=0
        previous_speaker = ''
        temp_end_time=0
        for i in final_lst_chat:
            if i['speaker']!=main_speaker:
                temp_end_time = i['end_time']
                previous_speaker = i['speaker']
            elif main_speaker!=previous_speaker and previous_speaker!='':
                if i['start_time']-temp_end_time>0:
                    if i['start_time']-temp_end_time<15:
                        response_time+=i['start_time']-temp_end_time
                    else:
                        pass

                previous_speaker = i['speaker']

        excellent_response_time = final_lst_chat[-1]['end_time']/15
        good_response_time = final_lst_chat[-1]['end_time']/8
        average_response_time = final_lst_chat[-1]['end_time']/6
        bad_response_time = final_lst_chat[-1]['end_time']/6

        if response_time<excellent_response_time:
            return {
                'status':'Excellent',
                'overall_delay_time_in_seconds':response_time,
                'response_time_in_seconds':final_lst_chat[-1]['end_time']-response_time,
                'overall_time_in_seconds':final_lst_chat[-1]['end_time']
            }
        elif response_time<good_response_time:
             return {
                'status':'Good',
                'overall_delay_time_in_seconds':response_time,
                'response_time_in_seconds':final_lst_chat[-1]['end_time']-response_time,
                'overall_time_in_seconds':final_lst_chat[-1]['end_time']
            }
        elif response_time<average_response_time:
             return {
                'status':'Average',
                'overall_delay_time_in_seconds':response_time,
                'response_time_in_seconds':final_lst_chat[-1]['end_time']-response_time,
                'overall_time_in_seconds':final_lst_chat[-1]['end_time']
            }
        else:
            return {
                'status':'Poor',
                'overall_delay_time_in_seconds':response_time,
                'response_time_in_seconds':final_lst_chat[-1]['end_time']-response_time,
                'overall_time_in_seconds':final_lst_chat[-1]['end_time']
            }

    def get_transcribe_chat(self):
        # Function to extract transcribed chat segments
        lst=[]
        for time in self.lst_time:
            start = int(time['start_time'])
            end = round(time['end_time'])
            print(f'start={start}')
            print(f'end={end}')
            print(f"speaker={time['label']}")

            if start!=end and time['label'] in self.lst_speaker:
                chunk = self.audio[start*self.sr:end*self.sr]
                r = self.speech_to_text_pipeline(chunk, generate_kwargs  = {"task":"transcribe", "language":"english"})
                print(f'Chat={r}')
                # r['text'] = self.remove_repeated_sequences(r['text'])
                lst.append({
                    'speaker':time['label'],
                    'text':r['text'],
                    'start_time':start,
                    'end_time':end
                })
                print(start,end,time['label'],r['text'])
        return lst

    def get_sentiment(self,text):
            # Function to perform sentiment analysis on a text
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            if sentiment_score >= 0:
                return "positive"
            elif sentiment_score < 0:
                return "negative"
            
            
    def get_main_speaker(self,lst):
        val=""
        speaker_name={''}
        count=0
        for data in lst:
            val+=f"{data['start_time']} {data['end_time']} {data['speaker']} {data['text']} "
            speaker_name.add(data['speaker'])
        speaker_name.pop()
        first_speaker=speaker_name.pop()
        second_speaker=speaker_name.pop()
        identify_speaker_input = {
                    'question': f'Who is seller {first_speaker}  or {second_speaker}?',
                    'context': val
                                }
        ans = self.identify_speaker_pipeline(identify_speaker_input)['answer']
    
        val=""
        speaker_name={''}
        try:
            ans = int(ans)
            for data in lst:
                if data['start_time']==ans or data['end_time']==ans:
                        return data['speaker']
        except:
            index = ans.lower().find("SPEAKER".lower())
            if index==-1:
                for data in lst:
                    if data['text'].lower().find(ans.lower()) !=-1:
                            return data['speaker']
            return ans[index:index+10]
        count+=1

    def preprocess_conversation(self,lst):
        # Function to preprocess the conversation text and merge adjacent chat segments from the same speaker
        lst_chat=[]
        prev_speaker=None

        for text in lst:
            if text['speaker']==prev_speaker:
                lst_chat[-1]['text']+= f" {self.remove_repeated_sequences(text['text'])}" 
                lst_chat[-1]['end_time']=text['end_time']

            else:
                lst_chat.append(text)

                prev_speaker=text['speaker']


        temp_lst_speakers=[]
        for i in lst_chat:
            temp_lst_speakers.append(i['speaker'])
        lst_speaker=[]
        for i in range(2):
            speaker_one = statistics.mode(temp_lst_speakers)
            lst_speaker.append(speaker_one)
            temp_lst_speakers = [speaker   for speaker in temp_lst_speakers if speaker!=speaker_one]
        lst_speaker
        final_lst_chat = []
        for i in lst_chat:
            if i['speaker']  in lst_speaker:
                final_lst_chat.append(i)

        main_speaker = self.get_main_speaker(final_lst_chat)
        return final_lst_chat,main_speaker

    def get_speaker_emotions(self,main_speaker):
        lst_emotions=[]
        for time  in self.lst_time:
            start = int(time['start_time'])
            end = round(time['end_time'])
        #     print(start,end)
            if start!=end:
                emotion_chunk = self.emotion_audio[start*self.emotion_sr:end*self.emotion_sr]
                if time['label']==main_speaker:
                    inputs = self.speech_emotion_feature_extractor(emotion_chunk, sampling_rate=16000, padding=True, return_tensors="pt")

                    logits = self.speech_emotion_model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    labels = [self.lstEmotions[_id] for _id in predicted_ids.tolist()][0]
                    if end-start>1:
                        lst_emotions.append(
                            labels
                        )
        return lst_emotions

    def remove_repeated_sequences(self,text):
            print('''----------Before Preprocessed---------''')
            print(text)
            print('''----------Done---------''')
            try:
                text = re.sub(r'[^\w\s]', '', text)
                # text = re.sub(r'\b(\s*\w+\s+)+(\w+\s*)+(\1)+\b', r'\2\1', text, flags=re.IGNORECASE)
                pattern = r'(\w)(\1{2,})|(\b\d+\b)(-\3)+'
                processed_text = re.sub(pattern, r'\1\3', text)
                merged_pattern = r'\b(\w+)\b(\.\s+\1\b)+'
                processed_text = re.sub(merged_pattern, r'\1.', processed_text)
            except:
                pass
            return processed_text

    def preprocess_test(self,lst_text):
            chat_lines=''
            prev_speaker=None
            print(lst_text)
            for text in lst_text:
                if text['speaker']==prev_speaker:
                    chat_lines+= f" {self.remove_repeated_sequences(text['text'])}"
                else:
                    chat_lines+=f"\n{text['speaker'] + ':' + self.remove_repeated_sequences(text['text'])}"
                    prev_speaker=text['speaker']
            # print(chat_lines[1:])
            print('----------All Done-----------------------')
            return chat_lines[1:]

        # Get the summary of the conversation using the summarization model
    def get_summary(self,lst_text):
        chat_lines=self.preprocess_test(lst_text)
        try:
            summary = self.summary_pipeline(chat_lines.lower())
        except:
            chat_lines = chat_lines[:2000]
            summary = self.summary_pipeline(chat_lines.lower())
        return summary
    
    def create_labelling(self):
        times = [((s.start + s.stop) / 2) / sampling_rate for s in self.wav_splits]
        labelling = []
        start_time = 0

        for i,time in enumerate(times):
            if i>0 and self.labels[i]!=self.labels[i-1]:
                temp = [str(self.labels[i-1]),start_time,time]
                labelling.append(tuple(temp))
                start_time = time
            if i==len(times)-1:
                temp = [str(self.labels[i]),start_time,time]
                labelling.append(tuple(temp))

        return labelling

    # Get the conversation text, preprocess it, and extract emotions and sentiments
    def get_conversation(self):
        self.lst_speaker=[]
        lst_max_speaker=[]
        self.lst_time =[]
        max_speaker=0
        max_speaker_name=''
        for j in range(2):
            for i in self.diarization.labels():
                if self.diarization.label_duration(i) > max_speaker and self.diarization.label_duration(i)  not in lst_max_speaker:
                    max_speaker=self.diarization.label_duration(i)
                    max_speaker_name=i
            self.lst_speaker.append(max_speaker_name)
            lst_max_speaker.append(max_speaker)
            max_speaker=0
        if lst_max_speaker[0]-lst_max_speaker[1]<100:
            prev_speaker=''  
            for segment, _, label in self.diarization.itertracks(yield_label=True):
                if label not in self.lst_speaker:
                    continue
                elif prev_speaker=='' and label in self.lst_speaker:
                    self.lst_time.append({
                        'start_time':segment.start,
                        'end_time':segment.end,
                        'label':label
                    })
                    prev_speaker=label
                elif prev_speaker==label:
                    self.lst_time[-1]['end_time']=segment.end
                    prev_speaker=label
                elif prev_speaker!=label and self.lst_time[-1]['end_time']<segment.end and label in self.lst_speaker:
                    self.lst_time.append({
                        'start_time':segment.start,
                        'end_time':segment.end,
                        'label':label
                    })
                    prev_speaker=label
        else:
            _, self.cont_embeds, self.wav_splits = self.encoder.embed_utterance(self.emotion_audio, return_partials=True, rate=16)
            self.labels = self.clusterer.predict(self.cont_embeds)
            labelling = self.create_labelling()
            self.lst_speaker=[]
         
            for i in range(2):
                self.lst_speaker.append(statistics.mode([f'SPEAKER_0{label}' for label,start,end in labelling if f'SPEAKER_0{label}' not in self.lst_speaker]))
            prev_speaker=''
            for label,start,end in labelling:
                if label not in self.lst_speaker:
                    continue
                if end-start<1:
                    continue
                if prev_speaker=='' and label in self.lst_speaker:
                    data={
                        'start_time':start,
                        'end_time':end,
                        'label':f'Speaker-{label}'
                    }
                    self.lst_time.append(data)
                    prev_speaker= label

                elif prev_speaker== label:
                    self.lst_time[-1]['end_time']= end
                    prev_speaker= label
                elif prev_speaker!= label and self.lst_time[-1]['end_time']<  end and label in self.lst_speaker:
                    data={
                        'start_time':start,
                        'end_time':end,
                        'label':f'Speaker-{label}'
                    }
                    self.lst_time.append(data)
                    prev_speaker= label
            
            
            
            
            
            
       
         
            
            
        lst=self.get_transcribe_chat()
        temp_lst = lst.copy()
        final_lst_chat,main_speaker = self.preprocess_conversation(temp_lst) 
        lst_emotions = self.get_speaker_emotions(main_speaker)
        lst_sentiment = []
        for i in final_lst_chat:
            if i['speaker']==main_speaker:
                lst_sentiment.append(self.get_sentiment(i['text']))
                
        return {
            'final_lst_chat':final_lst_chat,
            'lst_emotions':lst_emotions,
            'lst_sentiment':lst_sentiment,
            'main_speaker':main_speaker
        }

