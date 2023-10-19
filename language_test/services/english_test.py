import string
import language_tool_python
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import spacy
from sentence_transformers import SentenceTransformer
import torch
nlp = spacy.load("en_core_web_lg")
import librosa
from nltk.tokenize import word_tokenize
import textstat

class EnglishTestQuestion():
    
    def __init__(self, questions, correct_output, file_path,qa_pipeline,device,speech_to_text_pipeline,cefr_pipeline):
        
        
        # Initialize the question-answering pipeline
        self.qa_pipeline = qa_pipeline
        
        self.device = device
        
        # Initialize the automatic speech recognition pipeline
        self.speech_to_text_pipeline  = speech_to_text_pipeline
        
        self.context = ""
        self.file_path = file_path
        self.questions = questions
        self.correct_output = correct_output
        
        self.punctuation = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', 
                            ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
        
        # Extract context from speech using the speech-to-text pipeline
        self.context = self.speech_to_text()
        self.context = self.context['text']
        self.word_per_minute = self.get_word_per_minute()
        self.levels =  {
                'beginner': range(0, 9),
                'elementary': range(9, 11),
                'intermediate': range(11, 13),
                'upper-intermediate': range(13, 16),
                'advanced': range(16, 20),
                'proficiency':range(21,22)
            }
        self.lst_difficult_words=textstat.difficult_words_list(self.context)
        self.cefr_pipeline=cefr_pipeline

    def get_audio_duration(self):
        # Load the audio file with librosa to get the duration
        y, sr = librosa.load(self.file_path)

        # Get the audio duration in seconds using librosa
        duration_sec = librosa.get_duration(y=y, sr=sr)

        # Get the audio duration in minutes and seconds
        duration_min = int(duration_sec // 60)
        duration_sec = int(duration_sec % 60)

        return f"{duration_min} Min {duration_sec} Sec"

    def speech_to_text(self):
        audio,_ = librosa.load(self.file_path)
        return self.speech_to_text_pipeline(audio,generate_kwargs={'task':'transcribe','language':'english'})
    
    def remove_punctuation(self,text):
        # Create a translation table using the string.punctuation characters
        translator = str.maketrans('', '', string.punctuation)
        
        # Use the translation table to remove punctuation from the text
        text_without_punctuation = text.translate(translator)
        
        return text_without_punctuation

    def get_word_per_minute(self):
        audio,sample_rate = librosa.load(self.file_path) 
        duration = librosa.get_duration(y=audio,sr=sample_rate)/60
        word_count = len(word_tokenize(self.remove_punctuation(self.context)))
        return int(word_count/duration)

    def validate_answer(self):
        # Validate answers to questions based on the context
        for question in self.questions:
            result = self.qa_pipeline(question=question, context=self.context)
            if result['score'] > 0.01:
                print(f'Question: {question}')
                print(f"Answer: {result['answer']}")
            else:
                print(f'Answer to {question} not found in context')
   
    def check_grammar_vocabulary(self):
        # Check grammar and vocabulary errors in the context using language_tool_python
        tool = language_tool_python.LanguageTool('en-US')  
        grammar_errors = tool.check(self.context)
        lstErrors=[]
        for error in grammar_errors:
            lstErrors.append(error)
            # print("Error:", error)
            # print("Suggestions:", error.replacements)
            # print()
        lst =[]
        for i in lstErrors:
            lst.append({
                'context':i.context,
                'issue':i.ruleIssueType,
                'start':i.offset,
                'end':i.offset+i.errorLength,
                'suggestion':i.replacements
            })
            # print(i.context)
            # print(i.ruleIssueType)
            # print(i.offset)
            # print(i.offset+i.errorLength)
            # print(i.replacements)
        return lst
            
    def calculate_sentence_similarity(self):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate sentence embeddings
        user_embedding = model.encode([self.remove_punctuation(self.context)])
        correct_embedding = model.encode([self.remove_punctuation(val) for val in self.correct_output])
        similarity_score=[]
        # Calculate cosine similarity
        
        for embedding in correct_embedding:
            similarity_score.append(cosine_similarity(embedding.reshape(1,embedding.shape[0]), user_embedding)[0][0] * 100)
        
        return similarity_score
    
        

    # Function to determine the difficulty level of a word using WordNet synsets
    # def get_word_level(self,word):
    #     stop_words =  set(stopwords.words('english'))
    #     if word not in stop_words:
    #         synsets = wordnet.synsets(word)
    #         if synsets:
    #             max_depth = max(synset.max_depth() for synset in synsets)
    #             for level, depth_range in self.levels.items():
    #                 if max_depth in depth_range:
    #                     if word in self.lst_difficult_words and max_depth<9:
    #                         level = 'upper-intermediate'
    #                     if word  not in self.lst_difficult_words and max_depth>12:
    #                         level = 'beginner'
                        
    #                     return level
    #         return None
    #     return None

    def get_word_level(self,word):
        stop_words =  set(stopwords.words('english'))
        if word not in stop_words:
           levels = self.cefr_pipeline(word)
           indx = torch.argmax(torch.tensor([data['score'] for data in levels]))
           level =  levels[indx]['label']
           if level=='A1':
               return "beginner"
           elif level=='A2':
               return 'elementary'
           elif level=='B1':
               return "intermediate"
           elif level=='B2':
               return 'upper-intermediate'
           elif level=='C1':
               return 'advanced'
           elif level=='C2':
               return 'proficiency'
        
           
        return None

    # Analyze text and determine difficulty levels of words
    def classify_word(self,words):
        doc = nlp(words)
        level_words = {level: {''} for level in self.levels.keys()}
        level_words_percentage = {level: 0 for level in self.levels.keys()}
        count=0
        for token in doc:
            if not token.is_punct and not token.is_space:
                level = self.get_word_level(token.text.lower())
                if level is not None:
                    count+=1
                    level_words_percentage[level]+=1
                    
                    level_words[level].add(token.text.lower())
                    if list(level_words[level])[0]=='':
                        level_words[level].remove('')
                    
        if count!=0:           
            for level,_ in level_words_percentage.items():
                level_words_percentage[level] = (level_words_percentage[level]/count)*100
        return level_words,level_words_percentage

       

    def calculate_vocabulary_proficiency(self):
        #Calculate Proficiency Vocabulary lexical diversity in text
        words_in_context = word_tokenize(self.context.lower())
        words = words_in_context

        #remove basics words from context
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words and word not in self.punctuation]

        total_words_in_context = len(words_in_context)

        unique_words = set(words)
        total_unique_words = len(unique_words)

        lexical_diversity = total_unique_words / total_words_in_context

        fdist = FreqDist(words)
        most_common_words = fdist.most_common(10)

        # basic_word_count = 0
        # advanced_word_count = 0
        # word_count = 0
        
        level_words,level_words_percentage = self.classify_word(self.context)
        for i in level_words:
            level_words[i] =  list(level_words[i])
            

        # basic_word_percentage = (basic_word_count / word_count) * 100
        # advanced_word_percentage = (advanced_word_count / word_count) * 100

        return {
           "Most Common Words:": most_common_words,
           "Total Words:": total_words_in_context,
           "Total Unique Words:": total_unique_words,
           "Lexical Diversity:": lexical_diversity,
        #    "Basic Word Percentage:": basic_word_percentage,
        #    "Advanced Word Percentage:": advanced_word_percentage,
           "level_words":level_words,
           "level_words_percentage":level_words_percentage
        }
    