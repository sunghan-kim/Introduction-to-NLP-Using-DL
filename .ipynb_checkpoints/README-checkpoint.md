# Introduction-to-NLP-Using-DL

- [Wikidocs의 "딥 러닝을 이용한 자연어 처리 입문"](https://wikidocs.net/book/2155)의 내용을 학습한 Repository

<br>

# 목차

## Chapter 01. 자연어 처리(Natural Language Processing)란?

- v01. 아나콘다(Anaconda)와 코랩(Colab) - <font color="red"><b>생략</b></font>
- v02. 필요 프레임워크와 라이브러리 - <font color="blue"><b>완료</b></font>
- v03. 자연어 처리를 위한 NLTK와 KoNLPy 설치하기 - <font color="red"><b>생략</b></font>
- v04. 판다스(Pandas) and 넘파이(Numpy) and 맷플롭립(Matplotlib) - <font color="red"><b>생략</b></font>
- v05. 판다스 프로파일링(Pandas-Profiling) - <font color="blue"><b>완료</b></font>
- v06. 머신 러닝 워크플로우(Machine Learning Workflow) - <font color="blue"><b>완료</b></font>

<br>

## Chapter 02. 텍스트 전처리 (Text preprocessing)

- v01. 토큰화(Tokenization) - <font color="blue"><b>완료</b></font>
- v02. 정제(Cleaning) and 정규화(Normalization) - <font color="blue"><b>완료</b></font>
- v03. 어간 추출(Stemming) and 표제어 추출(Lemmatization) - <font color="blue"><b>완료</b></font>
- v04. 불용어(Stopword) - <font color="blue"><b>완료</b></font>
- v05. 정규 표현식(Regular Expression) - <font color="blue"><b>완료</b></font>
- v06. 정수 인코딩(Integer Encoding) - <font color="blue"><b>완료</b></font>
- v07. 원-핫 인코딩(One-hot encoding) - <font color="blue"><b>완료</b></font>
- v08. 단어 분리하기(Byte Pair Encoding, BPE) - <font color="blue"><b>완료</b></font>
- v09. 데이터의 분리(Splitting Data) - <font color="blue"><b>완료</b></font>

<br>

## Chapter 03. 언어 모델(Language Model)

- v01. 언어 모델(Language Model)이란? - <font color="blue"><b>완료</b></font>
- v02. 통계적 언어 모델(Statistical Language Model, SLM) - <font color="blue"><b>완료</b></font>
- v03. N-gram 언어 모델(N-gram Language Model) - <font color="blue"><b>완료</b></font>
- v04. 한국어에서의 언어 모델(Language Model for Korean Sentences) - <font color="red"><b>생략</b></font>
- v05. 펄플렉서티(Perplexity) - <font color="blue"><b>완료</b></font>
- v06. 조건부 확률(Conditional Probability) - <font color="blue"><b>완료</b></font>

<br>

## Chapter 04. 카운트 기반의 단어 표현(Count based word Representation)

- v01. 다양한 단어의 표현 방법
- v02. Bag of Words(BoW)
- v03. 문서 단어 행렬(Document-Term Matrix, DTM)
- v04. TF-IDF(Term Frequency-Inverse Document Frequency)

<br>

## Chapter 05. 문서 유사도(Document Similarity)

- v01. 코사인 유사도(Cosine Similarity)
- v02. 여러가지 유사도 기법

<br>

## Chapter 06. 토픽 모델링(Topic Modeling)

- v01. 잠재 의미 분석(Latent Semantic Analysis, LSA)
- v02. 잠재 디리클레 할당(Latent Dirichlet Allocation, LDA)
- v03. 잠재 디리클레 할당(LDA) 실습2

<br>

## Chapter 07. 머신 러닝(Machine Learning) 개요

- v01. 머신 러닝이란(What is Machine Learning?)
- v02. 머신 러닝 훑어보기
- v03. 선형 회귀(Linear Regression)
- v04. 로지스틱 회귀(Logistic Regression) - 이진 분류
- v05. 다중 입력에 대한 실습
- v06. 벡터와 행렬 연산
- v07. 소프트맥스 회귀(Softmax Regression) - 다중 클래스 분류

<br>

## Chapter 08. 딥 러닝(Deep Learning) 개요

- v01. 퍼셉트론(Perceptron)
- v02. 인공 신경망(Artificial Neural Network) 훑어보기
- v03. 딥 러닝의 학습 방법
- v03-4. 역전파(BackPropagation) 이해하기
- v04. 과적합(Overfitting)을 막는 방법들
- v05. 기울기 소실(Gradient Vanishing)과 폭주(Exploding)
- v06. 케라스(Keras) 훑어보기
- v06-7.케라스의 함수형 API(Keras Functional API)
- v07. 다층 퍼셉트론(MultiLayer Perceptron, MLP)으로 텍스트 분류하기
- v08. 피드 포워드 신경망 언어 모델(Neural Network Language Model, NNLM)

<br>

## Chapter 09. 순환 신경망(Recurrent Neural Network)

- v01. 순환 신경망(Recurrent Neural Network, RNN)
- v02. 장단기 메모리(Long Short-Term Memory, LSTM)
- v03. 게이트 순환 유닛(Gated Recurrent Unit, GRU)
- v04. RNN 언어 모델(Recurrent Neural Network Language Model, RNNLM)
- v05. RNN을 이용한 텍스트 생성(Text Generation using RNN)
- v06. 글자 단위 RNN(Char RNN)

<br>

## Chapter 10. 워드 임베딩(Word Embedding)

- v01. 워드 임베딩(Word Embedding)
- v02. 워드투벡터(Word2Vec)
- v03. 영어/한국어 Word2Vec 실습
- v04. 글로브(GloVe)
- v05. 사전 훈련된 워드 임베딩(Pre-trained Word Embedding)
- v06. 엘모(Embeddings from Language Model, ELMo)
- v07. 임베딩 벡터의 시각화(Embedding Visualization)

<br>

## Chapter 11. 텍스트 분류(Text Classification)

- v01. 케라스를 이용한 텍스트 분류 개요(Text Classification using Keras)
- v02. 스팸 메일 분류하기(Spam Detection)
- v03. 로이터 뉴스 분류하기(Reuters News Classification)
- v04. IMDB 리뷰 감성 분류하기(IMDB Movie Review Sentiment Analysis)
- v05. 나이브 베이즈 분류기(Naive Bayes Classifier)
- v06. 네이버 영화 리뷰 감성 분류하기(Naver Movie Review Sentiment Analysis)

<br>

## Chapter 12. 태깅 작업(Tagging Task)

- v01. 케라스를 이용한 태깅 작업 개요(Tagging Task using Keras)
- v02. 개체명 인식(Named Entity Recognition)
- v03. 양방향 LSTM을 이용한 개체명 인식(Named Entity Recognition using Bi-LSTM)
- v04. 양방향 LSTM을 이용한 품사 태깅(Part-of-speech Tagging using Bi-LSTM)
- v05. 양방향 LSTM과 CRF(Bidirectional LSTM + CRF)

<br>

## Chapter 13. 기계 번역(Neural Machine Translation)

- v01. 시퀀스-투-시퀀스(Sequence-to-Sequence, seq2seq)
- v02. 간단한 seq2seq 만들기(Simple seq2seq)
- v03. BLEU Score(Bilingual Evaluation Understudy Score)

<br>

## Chapter 14. 어텐션 메커니즘 (Attention Mechanism)

- v01. 어텐션 메커니즘 (Attention Mechanism)
- v02. 양방향 LSTM과 어텐션 메커니즘(BiLSTM with Attention mechanism)

<br>

## Chapter 15. 트랜스포머(Transformer)

- v01. 트랜스포머(Transformer)

<br>

## Chapter 16. 합성곱 신경망(Convolution Neural Network)

- v01. 합성곱 신경망(Convolution Neural Network)

