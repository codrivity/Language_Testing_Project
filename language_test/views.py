import json
from django.shortcuts import render
from django.http import HttpResponse
import torch
from django.views.decorators.csrf import csrf_exempt
from .services.dutch_test import DutchTestQuestion
from .services.english_test import EnglishTestQuestion
from .services.spanish_test import SpanishTestQuestion
from .services.english_conversation import EnglishConversationalSpeechtoText
import string
import os
from report.models import Report
from user.models import User
from transformers import pipeline
from transformers import (BartForConditionalGeneration,BartTokenizer, WhisperProcessor, 
                          WhisperForConditionalGeneration,AutoModelForQuestionAnswering,
                          AutoTokenizer,Wav2Vec2ForSequenceClassification,
                          Wav2Vec2FeatureExtractor,AutoModelForSequenceClassification)
from pyannote.audio import Pipeline
# Create your views here.

model_cache_dir = 'C:/Users/codri/Projects/models/'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
english_qa_model_name = "deepset/roberta-base-squad2"
spanish_qa_model_name="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
dutch_qa_model_name="henryk/bert-base-multilingual-cased-finetuned-dutch-squad2"
# spanish_speech_to_text_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
# dutch_speech_to_text_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-dutch"
english_conversation_emotion_model_name = "superb/wav2vec2-base-superb-er"
speech_to_text_model_whisper_name='openai/whisper-large-v2'
identify_speaker_model_name = 'deepset/deberta-v3-large-squad2'
identify_speaker_tokenizer = None
identify_speaker_model = None
identify_speaker_pipeline = None
english_qa_model = None
english_qa_tokenizer = None
english_qa_pipeline = None
spanish_qa_model = None
spanish_qa_tokenizer = None
spanish_qa_pipeline = None
dutch_qa_model = None
dutch_qa_tokenizer = None
dutch_qa_pipeline = None
speech_to_text_model = None
speech_to_text_processor = None
speech_to_text_pipeline = None
speaker_diarization_pipeline = None
summary_tokenizer = None
summary_model = None
summary_pipeline =  None
speech_to_text_chunck_size=30
speech_emotion_model =None
speech_emotion_feature_extractor = None
cefr_tokenizer=None
cefr_model=None
cefr_pipeline=None



speech_to_text_model = WhisperForConditionalGeneration.from_pretrained(speech_to_text_model_whisper_name,cache_dir=model_cache_dir)
speech_to_text_processor= WhisperProcessor.from_pretrained(speech_to_text_model_whisper_name,cache_dir=model_cache_dir)
speech_to_text_pipeline = pipeline(
    'automatic-speech-recognition',
    model=speech_to_text_model,
    tokenizer=speech_to_text_processor.tokenizer,
    feature_extractor=speech_to_text_processor.feature_extractor,
    chunk_length_s=speech_to_text_chunck_size,
    device=device
)

def remove_punctuation(text):
    # Create a translation table using the string.punctuation characters
    translator = str.maketrans('', '', string.punctuation)
    
    # Use the translation table to remove punctuation from the text
    text_without_punctuation = text.translate(translator)
    
    return text_without_punctuation


def handle_uploaded_file(f):
    lst_files = os.listdir('language_test/files')
    lst_files = [int(file.split('.')[0]) for file in lst_files]
    path='language_test/files/'
    if lst_files==[]:
        file_path=f'{path}1.wav'
    else:
        file_path = f'{path}{lst_files[-1]+1}.wav'
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

@csrf_exempt
def english_test_get_answer(request):
    if request.method == 'POST' and 'file' in request.FILES:
        
        uploaded_file = request.FILES['file']
        print(uploaded_file)
        # Process or save the file as required
        file_path = handle_uploaded_file(uploaded_file)
        print(file_path)
        #initialize variable globally
        global speech_to_text_chunck_size,device,english_qa_model_name,english_qa_model,english_qa_tokenizer,english_qa_pipeline,speech_to_text_pipeline

        #Check Variable is None or not
        if any(var is None for var in [speech_to_text_chunck_size, device, english_qa_model_name, english_qa_model, english_qa_tokenizer, english_qa_pipeline, speech_to_text_pipeline]):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            #initialize Question-Answer Pipeline
            english_qa_model_name = "deepset/roberta-base-squad2"
            english_qa_model = AutoModelForQuestionAnswering.from_pretrained(english_qa_model_name,cache_dir=model_cache_dir)
            english_qa_tokenizer = AutoTokenizer.from_pretrained(english_qa_model_name,cache_dir=model_cache_dir)
            english_qa_pipeline = pipeline("question-answering", model=english_qa_model, tokenizer=english_qa_tokenizer)
            #initialize Speech-to-text Pipeline
            
        cefr_tokenizer = AutoTokenizer.from_pretrained("hafidikhsan/distilbert-base-uncased-english-cefr-lexical-evaluation-bs-v3",cache_dir=model_cache_dir)
        cefr_model = AutoModelForSequenceClassification.from_pretrained("hafidikhsan/distilbert-base-uncased-english-cefr-lexical-evaluation-bs-v3",cache_dir=model_cache_dir,torch_dtype=torch.bfloat16).to(device)
        cefr_pipeline = pipeline(
        'text-classification',
            model=cefr_model,
            tokenizer=cefr_tokenizer,
            device=device,
            torch_dtype=torch.bfloat16
        )  
        #initialize a EnglishTestQuestion Class Object
        english_test = EnglishTestQuestion(questions=[],correct_output=['''
                                                Please call Stella.  Ask her to bring these things with her from the store:  
                                                Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a 
                                                snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  
                                                She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.
                                                               '''],file_path=file_path,qa_pipeline=english_qa_pipeline,device=device,speech_to_text_pipeline=speech_to_text_pipeline,cefr_pipeline=cefr_pipeline)
        

        transcribed_text = english_test.context['text']
        if not transcribed_text or len(transcribed_text.split()) <= 6:
            # Return an error response indicating that the user didn't speak enough
            return HttpResponse("User did not speak enough.")
        #return response
        response = {
            'context':english_test.context,
            'similarity_score':english_test.calculate_sentence_similarity(),
            'vocabulary_proficiency':english_test.calculate_vocabulary_proficiency(),
            'word_per_minute':english_test.word_per_minute,
            'duration':english_test.get_audio_duration(),
            'grammar_mistakes':english_test.check_grammar_vocabulary()
        }

        #remove audio file
        os.remove(file_path)

        response = json.dumps(response)
        user_instance = User.objects.get(id=request.POST['user_id'])
        report = Report(user_id=user_instance,details=json.loads(response),name=request.POST['report_name'])
        report.save()
        # Return a success response
        return HttpResponse(response, content_type='application/json')
    
    # Return an error response if no file was uploaded
    return HttpResponse("No file uploaded.")

@csrf_exempt
def spanish_test_get_answer(request):
    
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        # Process or save the file as required
        file_path = handle_uploaded_file(uploaded_file)

        #initialize variable globally
        global device,spanish_qa_model_name,spanish_qa_model,spanish_qa_tokenizer,spanish_qa_pipeline,speech_to_text_pipeline

        #Check Variable is None or not
        if any(var is None for var in [ device, spanish_qa_model_name, spanish_qa_model, spanish_qa_tokenizer, spanish_qa_pipeline, speech_to_text_pipeline]):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            #initialize Question-Answer Pipeline
            spanish_qa_model_name="mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es"
            spanish_qa_model = AutoModelForQuestionAnswering.from_pretrained(spanish_qa_model_name,cache_dir=model_cache_dir)
            spanish_qa_tokenizer = AutoTokenizer.from_pretrained(spanish_qa_model_name,cache_dir=model_cache_dir)
            spanish_qa_pipeline = pipeline("question-answering", model=spanish_qa_model, tokenizer=spanish_qa_tokenizer)
            #initialize Speech-to-text Pipeline
            # speech_to_text_model = Wav2Vec2ForCTC.from_pretrained(speech_to_text_model_whisper_name,cache_dir=model_cache_dir)
            # speech_to_text_processor= Wav2Vec2Processor.from_pretrained(speech_to_text_model_whisper_name,cache_dir=model_cache_dir)
            
       #initialize a SpanishTestQuestion Class Object
        spanish_test = SpanishTestQuestion(questions=[],correct_output=['''
                                                Durante nuestra conversación advertí que la multitud aumentaba, apretándose más. 
                                                espontáneamente venidas por uno de esos llamamientos morales, íntimos, 
                                                misteriosos, informulados, que no parten de ninguna voz oficial
                                                . Las siguientes cartas, supliendo 
                                                ventajosamente mi narración, me permitirán descansar un poco. 
                                                Esta inmensidad de la creación sólo favorece 
                                                a los pillos, que siempre encuentran donde ocultar el fruto de sus rapiñas.
                                                Entonces, y en la famosa mañanade que me ocupo, no estaba mi ánimo para consideraciones de tal índole,
                                                mucho menos en presencia de un conflicto popular que de minuto en minuto tomaba proporciones graves.
                                                               '''],file_path=file_path,qa_pipeline=spanish_qa_pipeline,speech_to_text_pipeline=speech_to_text_pipeline)
       
        transcribed_text = spanish_test.context['text']
        if not transcribed_text or len(transcribed_text.split()) <= 6:
            # Return an error response indicating that the user didn't speak enough
            return HttpResponse("User did not speak enough.")
        #return response
        response = {
            'context':spanish_test.context,
            'similarity_score':spanish_test.calculate_sentence_similarity(),
            'vocabulary_proficiency':spanish_test.calculate_vocabulary_proficiency(),
            'word_per_minute':spanish_test.word_per_minute,
            'duration':spanish_test.get_audio_duration(),
            'grammar_mistakes':spanish_test.check_grammar_vocabulary()
        }

        os.remove(file_path)
        response = json.dumps(response)
        # Return a success response
        return HttpResponse(response, content_type='application/json')
    
    # Return an error response if no file was uploaded
    return HttpResponse("No file uploaded.")




def dutch_test_get_answer(request):
    
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        # Process or save the file as required
        file_path = handle_uploaded_file(uploaded_file)

        #initialize variable globally
        global device,dutch_qa_model_name,dutch_qa_model,dutch_qa_tokenizer,dutch_qa_pipeline

        #Check Variable is None or not
        if any(var is None for var in [ device, dutch_qa_model_name, dutch_qa_model, dutch_qa_tokenizer, dutch_qa_pipeline]):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            #initialize Question-Answer Pipeline
            dutch_qa_model_name="henryk/bert-base-multilingual-cased-finetuned-dutch-squad2"
            dutch_qa_model = AutoModelForQuestionAnswering.from_pretrained(dutch_qa_model_name,cache_dir=model_cache_dir)
            dutch_qa_tokenizer = AutoTokenizer.from_pretrained(dutch_qa_model_name,cache_dir=model_cache_dir)
            dutch_qa_pipeline = pipeline("question-answering", model=dutch_qa_model, tokenizer=dutch_qa_tokenizer)
            #initialize Speech-to-text Pipeline
           
        #initialize a DutchTestQuestion Class Object
        dutch_test = DutchTestQuestion(questions=[],correct_output=['''
                                               Sinterklaas is een klassiek feest dat kinderen leuk vinden. De wet was actief geworden.
                                                We zijn geen slechte mensen omdat we niet volgens de wet werken. We voelden dat iets ons volgde door de bossen.
                                                Op de een of andere manier dansen de letters op de pagina. Deze openhaard is Victoriaans.
                                                Ze knipte haar vingernagels. Deze wollen trui is kriebelig.
                                                Al deze vrouwen zijn valsspelers, laat je niet voor de gek houden.
                                                De geest lachte op een vreemde manier. Laura pakte een sigaret en plaatste die in haar mond.
                                                Het gebouw schudde nog na door de aardbeving. De criminelen treiterden de politieagenten.
                                                Alle buren konden het horen als Jan seks had met zijn vriendin. De verpakking was kapot.
                                                               '''],file_path=file_path,qa_pipeline=dutch_qa_pipeline,speech_to_text_pipeline=speech_to_text_pipeline)
        
        transcribed_text = dutch_test.context['text']
        if not transcribed_text or len(transcribed_text.split()) <= 6:
            # Return an error response indicating that the user didn't speak enough
            return HttpResponse("User did not speak enough.")
        #return response
        response = {
            'context':dutch_test.context,
            'similarity_score':dutch_test.calculate_sentence_similarity(),
            'vocabulary_proficiency':dutch_test.calculate_vocabulary_proficiency(),
            'word_per_minute':dutch_test.word_per_minute,
            'duration':dutch_test.get_audio_duration(),
            'grammar_mistakes':dutch_test.check_grammar_vocabulary()
        }

        os.remove(file_path)
        response = json.dumps(response)
        # Return a success response
        return HttpResponse(response, content_type='application/json')
    
    # Return an error response if no file was uploaded
    return HttpResponse("No file uploaded.")

@csrf_exempt
def english_conversation_speech_to_text(request):
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        # Process or save the file as required
        file_path = handle_uploaded_file(uploaded_file)

        
        #initialize variable globally
        global speech_to_text_chunck_size,speaker_diarization_pipeline,device,summary_model,summary_tokenizer,summary_pipeline,speech_emotion_model,speech_emotion_feature_extractor,identify_speaker_model,identify_speaker_tokenizer,identify_speaker_pipeline

        #Check Variable is None or not
        if any(var is None for var in [speech_to_text_chunck_size,speaker_diarization_pipeline,device,summary_model,summary_tokenizer,summary_pipeline,speech_emotion_model,speech_emotion_feature_extractor,identify_speaker_model,identify_speaker_tokenizer,identify_speaker_pipeline]):
            print('.......................Something is None..................')
            speaker_diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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",cache_dir=model_cache_dir)
            hparams = speaker_diarization_pipeline.parameters(instantiated=True)
            hparams["segmentation"]["threshold"] =0.2442333667381752
            speaker_diarization_pipeline.instantiate(hparams)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            summary_tokenizer = BartTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum",cache_dir=model_cache_dir)
            summary_model = BartForConditionalGeneration.from_pretrained("philschmid/bart-large-cnn-samsum",cache_dir=model_cache_dir)
            summary_pipeline =  pipeline("summarization", model=summary_model, tokenizer=summary_tokenizer)
            

            #initialize identify speaker pipeline
            identify_speaker_tokenizer = AutoTokenizer.from_pretrained(identify_speaker_model_name, cache_dir=model_cache_dir)
            identify_speaker_model = AutoModelForQuestionAnswering.from_pretrained(identify_speaker_model_name, cache_dir=model_cache_dir)

            identify_speaker_pipeline = pipeline('question-answering', 
                                                model=identify_speaker_model, 
                                                tokenizer=identify_speaker_tokenizer,
                                                device=device)
            
            #initialize Speech-to-text Pipeline
            # speech_to_text_model = WhisperForConditionalGeneration.from_pretrained(speech_to_text_model,cache_dir=model_cache_dir,torch_dtype=torch.bfloat16).to(device)
            # speech_to_text_processor= WhisperProcessor.from_pretrained(speech_to_text_model_whisper_name,cache_dir=model_cache_dir)
            # speech_to_text_pipeline = pipeline(
            #     'automatic-speech-recognition',
            #     model=speech_to_text_model,
            #     tokenizer=speech_to_text_processor.tokenizer,
            #     feature_extractor=speech_to_text_processor.feature_extractor,
            #     chunk_length_s=speech_to_text_chunck_size,
            #     device=device
            #)
            speech_emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(english_conversation_emotion_model_name, cache_dir=model_cache_dir)
            speech_emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(english_conversation_emotion_model_name, cache_dir=model_cache_dir)
            
        #initialize a EnglishConversationSpeechToText Class Object
        englishConversationalSpeechtoText = EnglishConversationalSpeechtoText(file_name=file_path,speaker_diarization_pipeline=speaker_diarization_pipeline,
                                                                              summary_pipeline=summary_pipeline,speech_to_text_pipeline=speech_to_text_pipeline,
                                                                              speech_emotion_model=speech_emotion_model,speech_emotion_feature_extractor=speech_emotion_feature_extractor,
                                                                              identify_speaker_pipeline=identify_speaker_pipeline)
        conversation = englishConversationalSpeechtoText.get_conversation()
        
        os.remove(file_path)
        #return response
        response = {
            'conversation':conversation,
            'summary':englishConversationalSpeechtoText.get_summary(conversation['final_lst_chat'])[0],
            'response':englishConversationalSpeechtoText.get_response_time(conversation['final_lst_chat'],conversation['main_speaker'])
        }

        response = json.dumps(response)
        # Return a success response
        return HttpResponse(response, content_type='application/json')
    
    # Return an error response if no file was uploaded
    return HttpResponse("No file uploaded.")
