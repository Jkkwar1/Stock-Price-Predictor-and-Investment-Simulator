"""
CS152 Project: Stock Price Predictor and Investment Simulator
Author: Joseph-Richard Kwarteng
Date: 12/8/2023
Section: CS152
"""

from tkinter import *
from tkinter import ttk, messagebox
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

class Stock:
    """
    Represents a stock and provides methods for data handling and simulations.
    """

    def __init__(self, symbol, start_date, end_date):
        """
        Initializes a Stock instance with symbol, start_date, and end_date.

        Parameters:
        - symbol (str): Stock symbol.
        - start_date (str): Start date for data retrieval.
        - end_date (str): End date for data retrieval.
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None  # Placeholder for stock data

    def download_data(self):
        """
        Downloads stock data using yfinance and assigns it to the 'data' attribute.
        """
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)

    def calculate_daily_returns(self):
        """
        Calculates daily returns based on the 'Close' column of stock data.
        """
        # Check if 'Close' column exists
        if 'Close' not in self.data.columns:
            raise ValueError("No 'Close' column found in stock data.")

        # Check if there is sufficient data to calculate daily returns
        if len(self.data) <= 1:
            raise ValueError("Insufficient data to calculate daily returns.")

        # Calculate daily returns
        self.data['Daily_Return'] = self.data['Close'].pct_change().fillna(0)

    def run_monte_carlo_simulation(self, num_simulations, num_days):
        """
        Runs a Monte Carlo simulation to generate simulated stock prices.

        Parameters:
        - num_simulations (int): Number of simulations to run.
        - num_days (int): Number of days for each simulation.

        Returns:
        - np.ndarray: Simulated stock prices.
        """
        if self.data is not None:
            mu = self.data['Daily_Return'].mean()
            sigma = self.data['Daily_Return'].std()

            mu_simulations = np.random.normal(mu, sigma, num_simulations)
            sigma_simulations = np.random.normal(0, 1, num_simulations)

            simulated_prices = np.zeros((num_simulations, num_days))

            for i in range(num_simulations):
                for j in range(1, num_days):
                    simulated_prices[i, j] = simulated_prices[i, j - 1] * np.exp(
                        (mu_simulations[i] - 0.5 * sigma**2) + sigma_simulations[i] * np.sqrt(j))

            return simulated_prices

class Investment(Stock):
    """
    Represents an investment, inheriting from the Stock class.
    Provides methods for investment simulations.
    """

    def invest(self, simulation_index, num_shares):
        """
        Simulates an investment and returns the total value.

        Parameters:
        - simulation_index (int): Index of the simulation.
        - num_shares (float): Number of shares to invest.

        Returns:
        - float: Total value of the investment.
        """
        # Placeholder for investment simulation logic
        return np.random.uniform(0, 1) * num_shares * self.data['Close'].iloc[-1]

class StockPricePredictorApp:
    """
    Tkinter application for Stock Price Prediction and Investment Simulation.
    """

    def __init__(self, root):
        """
        Initializes the StockPricePredictorApp.

        Parameters:
        - root: Tkinter root window.
        """
        self.root = root
        self.root.geometry('800x600')
        self.root.title('Stock Price Predictor')

        # Create a notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=1, fill='both')

        # Prediction tab
        self.prediction_frame = Frame(self.notebook)
        self.notebook.add(self.prediction_frame, text='Stock Prediction')

        # Investments tab
        self.investments_frame = Frame(self.notebook)
        self.notebook.add(self.investments_frame, text='Investments')

        # Add a Label to display investment details
        self.investment_details_label = Label(self.investments_frame, text='')
        self.investment_details_label.pack()

        # Initialize content for each tab
        self.setup_prediction_tab()
        self.setup_investments_tab()

    def setup_prediction_tab(self):
        """
        Set up UI components for the Stock Prediction tab.
        """
        label = Label(self.prediction_frame, text='Stock Price Prediction')
        label.pack()

        # Entry boxes for prediction
        label_symbol = Label(self.prediction_frame, text='Symbol:')
        label_symbol.pack()
        self.entry_symbol = Entry(self.prediction_frame, width=10)
        self.entry_symbol.pack()

        label_start_date = Label(self.prediction_frame, text='Start Date:')
        label_start_date.pack()
        self.entry_start_date = Entry(self.prediction_frame, width=10)
        self.entry_start_date.pack()

        label_end_date = Label(self.prediction_frame, text='End Date:')
        label_end_date.pack()
        self.entry_end_date = Entry(self.prediction_frame, width=10)
        self.entry_end_date.pack()

        label_num_simulations = Label(self.prediction_frame, text='Num Simulations:')
        label_num_simulations.pack()
        self.entry_num_simulations = Entry(self.prediction_frame, width=10)
        self.entry_num_simulations.pack()

        label_num_days = Label(self.prediction_frame, text='Num Days:')
        label_num_days.pack()
        self.entry_num_days = Entry(self.prediction_frame, width=10)
        self.entry_num_days.pack()

        predict_button = Button(self.prediction_frame, text='Predict', command=self.predict_prices)
        predict_button.pack()

    def setup_investments_tab(self):
        """
        Set up UI components for the Investments tab.
        """
        label = Label(self.investments_frame, text='Investments Tab')
        label.pack()

        # Entry boxes for investments
        label_symbol_invest = Label(self.investments_frame, text='Symbol:')
        label_symbol_invest.pack()
        self.entry_symbol_invest = Entry(self.investments_frame, width=10)
        self.entry_symbol_invest.pack()

        label_start_date_invest = Label(self.investments_frame, text='Start Date:')
        label_start_date_invest.pack()
        self.entry_start_date_invest = Entry(self.investments_frame, width=10)
        self.entry_start_date_invest.pack()

        label_end_date_invest = Label(self.investments_frame, text='End Date:')
        label_end_date_invest.pack()
        self.entry_end_date_invest = Entry(self.investments_frame, width=10)
        self.entry_end_date_invest.pack()

        label_initial_investment = Label(self.investments_frame, text='Initial Investment:')
        label_initial_investment.pack()
        self.entry_initial_investment = Entry(self.investments_frame, width=10)
        self.entry_initial_investment.pack()

        label_num_shares = Label(self.investments_frame, text='Num Shares:')
        label_num_shares.pack()
        self.entry_num_shares = Entry(self.investments_frame, width=10)
        self.entry_num_shares.pack()

        invest_button = Button(self.investments_frame, text='Invest', command=self.invest)
        invest_button.pack()

    def predict_prices(self):
        """
        Retrieves input data and runs Monte Carlo simulation for stock price prediction.
        Displays the results in a plot.
        """
        # Get data from entry boxes and run Monte Carlo simulation
        symbol = self.entry_symbol.get()
        start_date = self.entry_start_date.get()
        end_date = self.entry_end_date.get()
        num_simulations = int(self.entry_num_simulations.get())
        num_days = int(self.entry_num_days.get())

        stock = Stock(symbol=symbol, start_date=start_date, end_date=end_date)
        stock.download_data()
        stock.calculate_daily_returns()
        simulated_prices = stock.run_monte_carlo_simulation(num_simulations=num_simulations, num_days=num_days)

        # Display the simulated prices in a plot
        fig, ax = plt.subplots(figsize=(6, 4))
        dates_actual = stock.data.index
        prices_actual = stock.data['Close']
        plt.plot(dates_actual, prices_actual, label=f"{symbol} - Actual Prices", linestyle='-')

        for i in range(num_simulations):
            dates_simulated = dates_actual[:num_days]  # Take only the first num_days elements
            prices_simulated = simulated_prices[i, :num_days]

            # Ensure the lengths match
            if len(dates_simulated) != len(prices_simulated):
                print(f"Simulation {i + 1}: Mismatched dimensions between dates and simulated prices. Skipping.")
            else:
                plt.plot(dates_simulated, prices_simulated, label=f"{symbol} - Simulated Prices {i + 1}", linestyle='--')

        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'{symbol} Stock Prices Over Time')
        plt.legend()
        plt.show()

    def invest(self):
        """
        Retrieves input data and simulates an investment.
        Displays investment details on the Tkinter app and in a message box.
        """
        # Get data from entry boxes and simulate investment
        symbol_invest = self.entry_symbol_invest.get()
        start_date_invest = self.entry_start_date_invest.get()
        end_date_invest = self.entry_end_date_invest.get()
        initial_investment = float(self.entry_initial_investment.get())
        num_shares = float(self.entry_num_shares.get())

        investment = Investment(symbol=symbol_invest, start_date=start_date_invest, end_date=end_date_invest)
        investment.download_data()
        investment.calculate_daily_returns()
        total_value = investment.invest(simulation_index=0, num_shares=num_shares)

        # Display the investment details on the Tkinter app
        investment_details = f"Symbol: {symbol_invest}, Initial Investment: {initial_investment}, " \
                             f"Number of Shares: {num_shares}, Total Value: {total_value}"
        self.investment_details_label.config(text=investment_details)

        # Display a message box with the investment details
        messagebox.showinfo("Investment Details", investment_details)

def main():
    """
    Main function to initialize the Tkinter app and run the main event loop.
    """
    root = Tk()
    app = StockPricePredictorApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
