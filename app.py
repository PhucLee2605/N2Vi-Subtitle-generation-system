from src import create_app
import warnings
warnings.filterwarnings("ignore")

app = create_app()

app.run(debug=True, port=5001)
