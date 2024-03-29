
# Sales Prediction for Big Mart Outlets

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.

Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.

Please note that the data may have missing values as some stores might not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

## Data Dictionary
We have a train (8523) and test (5681) data set, the train data set has both input and output variable(s). You need to predict the sales for the test data set.

## Train file:
CSV containing the item outlet information with a sales value

### Variable Description
ItemIdentifier ---- Unique product ID
ItemWeight ---- Weight of product
ItemFatContent ---- Whether the product is low fat or not
ItemVisibility ---- The % of the total display area of all products in a store allocated to the particular product
ItemType ---- The category to which the product belongs
ItemMRP ---- Maximum Retail Price (list price) of the product
OutletIdentifier ---- Unique store ID
OutletEstablishmentYear ---- The year in which the store was established
OutletSize ---- The size of the store in terms of ground area covered
OutletLocationType ---- The type of city in which the store is located
*OutletType ---- Whether the outlet is just a grocery store or some sort of supermarket
ItemOutletSales ---- sales of the product in t particular store. This is the outcome variable to be predicted.

## Test file:
CSV containing item outlet combinations for which sales need to be forecasted

### Variable Description
ItemIdentifier ----- Unique product ID
ItemWeight ---- Weight of product
ItemFatContent ----- Whether the product is low fat or not
ItemVisibility ---- The % of the total display area of all products in a store allocated to the particular product
ItemType ---- The category to which the product belongs
ItemMRP ----- Maximum Retail Price (list price) of the product
OutletIdentifier ----- Unique store ID
OutletEstablishmentYear ----- The year in which store store was established
OutletSize ----- The size of the store in terms of ground area covered
OutletLocationType ---- The type of city in which the store is located
OutletType ---- whether the outlet is just a grocery store or some sort of supermarket

### Submission file format
### Variable Description
ItemIdentifier ----- Unique product ID
OutletIdentifier ----- Unique store ID
ItemOutletSales ----- Sales of the product in t particular store. This is the outcome variable to be predicted.

### Evaluation Metric
Your model performance will be evaluated on the basis of your prediction of the sales for the test data (test.csv), which contains similar data-points as train except for the sales to be predicted. Your submission needs to be in the format as shown in the same sample submission.

We at our end, have the actual sales for the test dataset, against which your predictions will be evaluated. We will use the Root Mean Square Error value to judge your response.