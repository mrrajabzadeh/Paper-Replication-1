# Table 1 Results Replication

This project is focused on replicating the Table 1 results of the paper "Style investing, comovement and return predictability" by Sunil Wahal and M. Deniz Yavuz, published in the Journal of Financial Economics 107 (2013) 136–154. The code performs comprehensive data handling, processing, and regression analysis to explore stock style investing, comovement, and return predictability.

## Overview

The code aims to:
- Fetch and prepare stock price data from the CRSP database using SQL queries.
- Process data to calculate market equity and book-to-market ratios.
- Create portfolio returns based on quintiles for size and B/M ratio.
- Conduct Fama-MacBeth regression analyses to predict future returns based on past performance metrics and style returns.

## Features

- **Data Extraction**: Utilizes the WRDS database to retrieve stock data and delisting returns from the CRSP database for the period between 1965 and 2009.
- **Data Processing**: Filters data, merges multiple datasets, and calculates key financial indicators like market equity and book-to-market ratios.
- **Portfolio Returns Calculation**: Groups stocks into quintiles based on size and B/M ratio and calculates returns for these portfolios.
- **Regression Analysis**: Runs multiple Fama-MacBeth regressions using various combinations of past returns, style returns, and financial metrics as independent variables to predict future returns.

## Dependencies

To run this code, you need access to Python and the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `statsmodels`
- `wrds`
- `linearmodels`
- `scipy`

Additionally, you will need a valid WRDS account to fetch data from their databases.

## File Structure

- `replication_script.py`: Main Python script containing all the code to perform the data fetching, processing, regression analysis, and output generation.
- `CCM.csv`: An external CSV file containing Compustat-CRSP Merged (CCM) dataset links needed for the analysis.
- `regression_results_Final.txt`: Output file where the regression analysis results are saved.
- `cc.csv`: Final processed dataset saved after all calculations and before regression analysis.

## Usage

1. **Set up environment**: Ensure Python and all dependencies are installed, and you have a WRDS account set up.
2. **Run the script**: Execute the `replication_script.py` to perform the analysis.
3. **Review results**: Check the `regression_results_Final.txt` for regression outputs and `cc.csv` for the processed data.

## Contributing

Contributions to the replication and extension of this analysis are welcome. Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

