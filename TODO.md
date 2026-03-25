# TSLA Streamlit Deployment TODO

## Plan Steps (Approved)
- [x] 1. Install dependencies **OK** (matplotlib 3.10.8, numpy, etc. | tensorflow pending)
- [x] 2. Train models: `python src/train.py` **PARTIAL (h1/h5 OK, h10 pending)**
- [ ] 3. Verify models
- [ ] 4. Test `streamlit run app/streamlit_app.py`
- [ ] 5. Deploy to Streamlit Cloud (push models)

**Status:** Core deps ready. **Training RNN/LSTM models now** (6 models + gridsearch, ~20-40min CPU). 
**Output:** Watch terminal for progress (epochs/loss).

