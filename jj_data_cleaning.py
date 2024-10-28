# %%
import pandas as pd

# Load the datasets
menu_price_cost_df = pd.read_excel('MenuPriceCost.xlsx')
jj202301_df = pd.read_excel('jj202301.v1.xlsx')

# Unpivot the jj202301 data to long format
orders_df = pd.melt(jj202301_df, 
                    id_vars=['Date', 'Day of Week', 'Payment Type', 'Customer ID', 'Approval', 'Payment', 'Commision', 'Total VAT', 'Total Fee'],
                    value_vars=[f'Order {i}' for i in range(1, 11)],
                    var_name='Order_Num', value_name='Order')

prices_df = pd.melt(jj202301_df, 
                    id_vars=['Date', 'Day of Week', 'Payment Type', 'Customer ID', 'Approval', 'Payment', 'Commision', 'Total VAT', 'Total Fee'],
                    value_vars=[f'Price {i}' for i in range(1, 11)],
                    var_name='Price_Num', value_name='Price')

costs_df = pd.melt(jj202301_df, 
                   id_vars=['Date', 'Day of Week', 'Payment Type', 'Customer ID', 'Approval', 'Payment', 'Commision', 'Total VAT', 'Total Fee'],
                   value_vars=[f'Cost {i}' for i in range(1, 11)],
                   var_name='Cost_Num', value_name='Cost')

# Ensure the order, price, and cost data align by row
orders_df = orders_df.sort_values(by=['Date', 'Customer ID', 'Order_Num']).reset_index(drop=True)
prices_df = prices_df.sort_values(by=['Date', 'Customer ID', 'Price_Num']).reset_index(drop=True)
costs_df = costs_df.sort_values(by=['Date', 'Customer ID', 'Cost_Num']).reset_index(drop=True)

# Combine the data into a single dataframe
combined_df = orders_df[['Date', 'Day of Week', 'Payment Type', 'Customer ID', 'Approval', 'Order']].copy()
combined_df['Price'] = prices_df['Price']
combined_df['Cost'] = costs_df['Cost']
combined_df['Payment'] = orders_df['Payment']
combined_df['Commision'] = orders_df['Commision'] - 500
combined_df['Total VAT'] = orders_df['Total VAT']
combined_df['Total Fee'] = orders_df['Total Fee'] - 500

# Remove rows where Order is NaN
combined_df = combined_df.dropna(subset=['Order'])

# Count the number of orders per transaction
order_counts = combined_df.groupby(['Date', 'Customer ID']).size().reset_index(name='Order Count')

# Merge the order counts with the combined dataframe
combined_df = pd.merge(combined_df, order_counts, on=['Date', 'Customer ID'])

# Divide the Commision, Total VAT, and Total Fee equally among the orders
combined_df['Commision'] = combined_df['Commision'] / combined_df['Order Count']
combined_df['Total VAT'] = combined_df['Total VAT'] / combined_df['Order Count']
combined_df['Total Fee'] = combined_df['Total Fee'] / combined_df['Order Count']

# Calculate the correct revenue per order
combined_df['Revenue'] = combined_df['Payment'] - combined_df['Cost'] - combined_df['Commision'] - combined_df['Total VAT'] - combined_df['Total Fee']

# Drop the 'Order Count' and 'Payment' columns as they are no longer needed
combined_df = combined_df.drop(columns=['Order Count', 'Payment'])

# Merge with menu price and cost data
menu_price_cost_df.columns = ['Order', 'Menu Price', 'Menu Cost']
final_df = pd.merge(combined_df, menu_price_cost_df, on='Order', how='left')

final_df.head(15)
#Save the final combined data to a new Excel file
#final_combined_file_path = 'user/cmw/downloads/jj202301_long_combined_v5.xlsx'
#final_df.to_excel(final_combined_file_path, index=False)

#final_combined_file_path


# %%
import pandas as pd

# Assuming df is your DataFrame
final_df.to_excel('jjmerged.xlsx', index=False)

# To create a download link
from IPython.display import FileLink

# Create a download link
FileLink('jjmerged.xlsx')


