# Demo app

    pip install -r requirements-quantum.txt
    make baselines          # the app refuses to predict without measured metrics
    streamlit run app/streamlit_app.py

Four screens: predict, explain, benchmark, cost. Two rules are enforced in code rather
than left to discipline -- no prediction is shown without the model's sensitivity and
specificity beside it, and the benchmark screen states plainly when the
parameter-matched control is missing.
