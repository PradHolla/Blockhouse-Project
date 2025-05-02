# Blockhouse Project

## Project Overview
This project implements and analyzes Order Flow Imbalance (OFI) features for financial market analysis, based on the research paper "Cross-Impact of Order Flow Imbalance in Equity Markets." The implementation extracts valuable insights from limit order book data to understand price dynamics and predict market movements.

## Background
Order Flow Imbalance measures the net order flow pressure at different levels of the limit order book. These imbalances have been shown to significantly explain price movements over short time intervals. This project implements four types of OFI calculations:

**Best-Level OFI**: Measures imbalance at the best bid/ask level

**Multi-Level OFI**: Extends analysis to deeper levels of the order book

**Integrated OFI**: Combines multi-level OFIs using Principal Component Analysis

**Cross-Asset OFI**: Examines relationships between OFIs of different assets

## Data Format

The implementation works with limit order book data in CSV format. Each row represents an update to the order book with information about:

- Timestamps

- Order types (additions, cancellations, trades)

- Price levels

- Order sizes

- Bid/ask information at multiple levels

## Implementation Details
### Best-Level OFI
Calculates the accumulative OFIs at the best bid/ask level, aggregated over specified time intervals.

### Multi-Level OFI
Extends the best-level OFI analysis to deeper levels of the limit order book, providing a more comprehensive view of market dynamics.

### Integrated OFI
Uses Principal Component Analysis (PCA) to combine information from multiple levels of the order book into a single metric, capturing the most significant patterns in the data.

### Cross-Asset OFI
Implements LASSO regression to model the impact of OFIs from multiple assets on each other's returns, with a focus on identifying sparse cross-impact relationships.

## Running the Code
To run the code, enter the following command in your terminal:

```bash
python OFI.py
```


## Key Findings
From the conceptual analysis:

**Multi-Level Depth Significance**: Deeper levels of the order book contain valuable information that better explains price movements, with high-volume and low-volatility stocks showing more influence from deeper levels.

**LASSO vs OLS for Cross-Impact**: LASSO regression is preferred for cross-impact estimation due to its ability to handle dimensionality challenges, multicollinearity issues, and enforce sparsity assumptions.

**OFI vs Trade Volume**: OFI is a better predictor of short-term returns than traditional trade volume because it captures both executed trades and limit order placements/cancellations, providing a comprehensive view of supply and demand imbalances.