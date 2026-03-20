# Automated Marking Making - A comparisson study
Automated Market Makers (AMMs) provide liquidity to the market without reliance on a traditional order book. When figuring out the price of securities a popular approach is to utilize a __Cost Function__ $C$, which will determine the cost of holding a certain sets of securities. When pricing how much to sell a bundle of securities $r$ given we are holding $q$ (where $q$ is a $K$ dimentional vector, where each entry $q_i$ corresponds to how much of of security $i$ we are holding), the price will be $C(q + r) - C(q)$. 

  

## Quick Start
Our project is implemented in python3, in the project folder please
```bash
python3 -m venv venv
source venv/bin/activate
```