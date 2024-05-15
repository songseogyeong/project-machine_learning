<br></br>

<div>
  <hr>
    <div style='text-align: center'>
    <b>
      <p style='font-size: 30px'> Healthcare Diabetes Classifier Project</p> 
    </b>
    <span style='font-size: 25px'> 당뇨병 진단 예측</span>
    </div>
  <hr>
    <div style='text-align: center'>
      <br></br>
      <br></br>
      <br></br>
      <br></br>
      <br></br>
      <br></br>
      <br></br>
      <span style='font-size: 25px'>2024. 05. 15.</span>
      <br></br>
      <br></br>
      <br></br>
      <br></br>
      <span style='font-size: 20px'>코리아IT 아카데미 - 송서경</span>
    </div>
</div>

<br></br>
<br></br>
<br></br>
<hr>
<br></br>
<br></br>
<br></br>

<h1> □ 목차</h1>
<br>
<div>
    <a style="color: inherit;" href='#Ⅰ. 개요'>
      <p style='font-size: 23px'>Ⅰ. 개요</p> <br>
    </a>
    <p style='font-size: 20px; margin-left: 20px; margin-top: -30px;'>1. 목적</p>
    <p style='font-size: 20px; margin-left: 20px;'>2. 분석 방법</p>
    <br></br>
    <a style="color: inherit;" href='Ⅱ. 분석 결과'>
      <p style='font-size: 23px'>Ⅱ. 분석 결과</p> <br>
    </a>
      <p style='font-size: 20px; margin-left: 20px; margin-top: -30px;'>1. 데이터 탐색</p>
    <p style='font-size: 20px; margin-left: 20px;'>2. 데이터 전처리</p>
    <p style='font-size: 20px; margin-left: 20px;'>3. 훈련</p>
    <br></br>
    <a style="color: inherit;" href='Ⅲ. 평가'>
      <p style='font-size: 23px'>Ⅲ. 평가</p> <br>
    </a>
    <p style='font-size: 20px; margin-left: 20px; margin-top: -30px;'>1. 평가 및 개선방안</p>
</div>

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

<h1 id="Ⅰ. 개요">Ⅰ. 개요</h1>
<p style='font-size: 25px'>1. 데이터 정보</p>
<div style='margin-left: 20px; font-size: 16px;'>
  <p style='font-size: 23px'>□ 당뇨병 진단 데이터 세트</p>
  <div style='margin-left: 20px;'>
    <p style='font-size: 18px;'>○ 데이터 출처</p>
    <a herf="https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes">kaggle - Healthcare Diabetes Dataset</a>
  </div>

  <br></br>

  <div style='margin-left: 20px;'>
    <p style='font-size: 18px;'>○ 데이터 정보</p>
    - 파일이름: 당뇨병 진단 데이터 세트 <br>
    - 파일 형식: CSV 파일 <br>
    - 구분: 이진 분류 데이터 세트 <br>
    - 형태: 2768 rows × 10 columns
  </div>
</div>

<br></br>
<br></br>

<p style='font-size: 25px'>2. 목적</p>
<div style='margin-left: 20px; font-size: 16px;'>
  <p style='font-size: 23px'>□ 분석 목적</p>
  <p style='font-size: 18px; margin-left: 20px;'>○ 다양한 건강 지표와 당뇨병의 관계를 탐색하고 당뇨병 발생 여부를 분석함으로써 <br>&nbsp;&nbsp;&nbsp;&nbsp;당뇨병 발생 여부를 예측하는 모델을 생성</p>
  <p style='font-size: 18px; margin-left: 20px;'>○ 예측 모델을 통해 환자의 당뇨병 예방 및 관리의 질 향상</p>
</div>

<br></br>


<p style='font-size: 25px'>3. 분석 방법</p>
<div style='margin-left: 20px; font-size: 16px;'>
  <p style='font-size: 23px'>□ 분석 절차</p>
  <div style='margin-left: 20px;'>
    <p style='font-size: 18px;'>○ 분석 프로세스</p>
    - 데이터 분석은 총 4가지 과정을 거쳐 진행함.
    <br></br>
    <table>
      <tr>
          <td>데이터 탐색</td>
          <td>데이터 전처리</td>
          <td>데이터 훈련</td>
          <td>데이터 평가</td>
      </tr>
      <tr>
          <td>데이터 이해, 문제 인식 및 해결안 도출</td>
          <td>전처리를 통한 데이터의 질 향상</td>
          <td>데이터 훈련을 통해 예측 모델 구축</td>
          <td>데이터 품질 측정 및 개선</td>
      </tr>
    </table>
  </div>
</div>

<br></br>
<br></br>

<div style='margin-left: 20px; font-size: 16px;'>
  <p style='font-size: 23px'>□ 분석 방법</p>
  <div style='margin-left: 20px;'>
  <p style='font-size: 18px;'>○ 탐색</p>
  - 데이터 정보 확인을 통한 데이터 이해, 문제 인식 및 해결안 도출
  </div>

  <br></br>

  <div style='margin-left: 20px;'>
    <p style='font-size: 18px;'>○ 전처리</p>
    - 불필요한 데이터 제거, 결측치, 이상치 등 제거를 통한 데이터 질 향상
  </div>

  <br></br>

  <div style='margin-left: 20px;'>
    <p style='font-size: 18px;'>○ 훈련</p>
    - 데이터 훈련 모델을 통한 예측 모델 구축
    <table>
      <tr>
          <td>연번</td>
          <td>모델</td>
      </tr>
      <tr>
          <td>1</td>
          <td>LogisticRegression</td>
      </tr>
    </table>
  </div>
  
  <br></br>
  
  <div style='margin-left: 20px;'>
    <p style='font-size: 18px;'>○ 평가</p>
    - OLS, VIF 등 다양한 평가 지표를 통한 데이터 품질 측정 및 개선 <br>
    - 데이터 평가 점수 산출 방식
    <table>
        <tr>
            <td>연번</td>
            <td> 평가지표</td>
            <td>산출 코드</td>
        </tr>
        <tr>
            <td>1</td>
            <td>accuracy (정확도)</td>
            <td>accuracy_score(y_test , prediction)</td>
        </tr>
        <tr>
            <td>2</td>
            <td>precision (정밀도)</td>
            <td>precision_score(y_test , prediction)</td>
        </tr>
        <tr>
            <td>3</td>
            <td>recall (재현율)</td>
            <td>recall_score(y_test , prediction)</td>
        </tr>
        <tr>
            <td>4</td>
            <td>f1</td>
            <td>f1_score(y_test, prediction)</td>
        </tr>
        <tr>
            <td>5</td>
            <td>roc_auc</td>
            <td>roc_auc_score(y_test, prediction)</td>
        </tr>
    </table>
  </div>
</div>

<br></br>
<br></br>
<br></br>
<br></br>
<br></br>

<h1 id="Ⅱ. 분석 결과">Ⅱ. 분석 결과</h1>