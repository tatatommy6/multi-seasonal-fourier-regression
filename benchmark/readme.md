## Rule for benchmark MSFR
1. 벤치마크 용 데이터는 Kaggle, Huggingface 등에 배보된 오픈소스 데이터셋만 사용하기
2. 벤치마크 폴더(이 `readme.md` 파일이 존재하는 폴더)에 벤치마크마다 폴더 만들기
3. 어떤 데이터셋을 사용했는지 `readme.md`를 만들어 라이선스에 맞게 출처 남기기(Jupyter Notebook 사용 시 노트북 파일에 markdown을 사용하여 readme.md 파일을 대신해도 됨)
4. 상관은 없으나 되도록 아래 템플릿을 따라 작성하기 바람

### Template for benchmark
- readme.md

``` markdown
## {벤치마크 명} for MSFR - {MSFR 버전}
- Dataset : {데이터셋 링크} by {데이터셋 만든 유저명}
- Reason why we should do this Benchmark : {이 벤치마크 파일을 생성한 이유 (예: 시장 수요 데이터셋은 기본적으로 계절에 영향을 받아 주기성이 있을 것으로 판단, 또한 각 품목마다 그 주기가 다르고, 추세도 있을 확률이 있어 MSFR의 장점인 주기 & 비주기 동시 예측에 적합하다고 생각)}
- Tester : {벤치마크를 만든 사람, 진행할 사람}

{아래는 자유롭게 벤치마크 결과를 요약}
```
- Project Structure
```
{벤치마크 명의 폴더}
- Dataset
  - {데이터들}
- model.py
- preprocessing.py (or ipynb)
- train.py (or ipynb)
- eval.py (or ipynb)
- readme.md
{등등 기타 원하는 파일들}
```
