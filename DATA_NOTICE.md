# Data notice

This project uses market data from the **Binance Vision** public data archive,
<https://data.binance.vision>.

- The datasets are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International licence (CC BY-NC-SA 4.0). Use is governed by the Binance Vision Dataset Terms:
  <https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md>
  (v1.0, last updated 2026-08-26).
- **This repository does not host any Binance data**, raw or processed. `scripts/download_data.py`
  fetches the files directly from the archive and verifies each one against its published SHA-256
  checksum. The download date of every file is recorded in `data/manifest.csv`.
- Derived artefacts published from this project (processed data, figures, fitted model artefacts)
  credit Binance Vision and are licensed under CC BY-NC-SA 4.0.
- The source code is licensed under the MIT licence (see `LICENSE`). That licence covers the code
  only, not the data or the derived artefacts.
- This project is non-commercial research. It is not affiliated with or endorsed by Binance.
