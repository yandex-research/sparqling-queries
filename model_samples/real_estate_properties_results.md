 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **Ref_Feature_Types** | feature_type_code | + |   | 
 | 2 |   | feature_type_name |   |   | 
 | 3 | **Ref_Property_Types** | property_type_code | + |   | 
 | 4 |   | property_type_description |   |   | 
 | 5 | **Other_Available_Features** | feature_id | + |   | 
 | 6 |   | feature_type_code |   | --> 1 | 
 | 7 |   | feature_name |   |   | 
 | 8 |   | feature_description |   |   | 
 | 9 | **Properties** | property_id | + |   | 
 | 10 |   | property_type_code |   | --> 3 | 
 | 11 |   | date_on_market |   |   | 
 | 12 |   | date_sold |   |   | 
 | 13 |   | property_name |   |   | 
 | 14 |   | property_address |   |   | 
 | 15 |   | room_count |   |   | 
 | 16 |   | vendor_requested_price |   |   | 
 | 17 |   | buyer_offered_price |   |   | 
 | 18 |   | agreed_selling_price |   |   | 
 | 19 |   | apt_feature_1 |   |   | 
 | 20 |   | apt_feature_2 |   |   | 
 | 21 |   | apt_feature_3 |   |   | 
 | 22 |   | fld_feature_1 |   |   | 
 | 23 |   | fld_feature_2 |   |   | 
 | 24 |   | fld_feature_3 |   |   | 
 | 25 |   | hse_feature_1 |   |   | 
 | 26 |   | hse_feature_2 |   |   | 
 | 27 |   | hse_feature_3 |   |   | 
 | 28 |   | oth_feature_1 |   |   | 
 | 29 |   | oth_feature_2 |   |   | 
 | 30 |   | oth_feature_3 |   |   | 
 | 31 |   | shp_feature_1 |   |   | 
 | 32 |   | shp_feature_2 |   |   | 
 | 33 |   | shp_feature_3 |   |   | 
 | 34 |   | other_property_details |   |   | 
 | 35 | **Other_Property_Features** | property_id |   | --> 9 | 
 | 36 |   | feature_id |   | --> 5 | 
 | 37 |   | property_feature_description |   |   | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_dev_1030 | How many available features are there in total? | SELECT count(*) FROM Other_Available_Features |  | 1. SELECT[tbl:​Other_Available_Features] <br>2. COMPARATIVE[#1, #1, None] <br>3. AGGREGATE[count, #2] <br> | + | easy | 
  | SPIDER_dev_1031 | What is the feature type name of feature AirCon? | SELECT T2.feature_type_name FROM Other_Available_Features AS T1 JOIN Ref_Feature_Types AS T2 ON T1.feature_type_code  =  T2.feature_type_code WHERE T1.feature_name  =  "AirCon" |  | 1. SELECT[val:​Other_Available_Features:​feature_name:​AirCon] <br>2. PROJECT[col:​Ref_Feature_Types:​feature_type_name, #1] <br> | + | medium | 
  | SPIDER_dev_1032 | Show the property type descriptions of properties belonging to that code. | SELECT T2.property_type_description FROM Properties AS T1 JOIN Ref_Property_Types AS T2 ON T1.property_type_code  =  T2.property_type_code GROUP BY T1.property_type_code |  | 1. SELECT[tbl:​Ref_Property_Types] <br>2. PROJECT[col:​Ref_Property_Types:​property_type_description, #1] <br> | + | medium | 
  | SPIDER_dev_1033 | What are the names of properties that are either houses or apartments with more than 1 room? | SELECT property_name FROM Properties WHERE property_type_code  =  "House" UNION SELECT property_name FROM Properties WHERE property_type_code  =  "Apartment" AND room_count  >  1 |  | 1. SELECT[tbl:​Properties] <br>2. PROJECT[col:​Properties:​room_count, #1] <br>3. GROUP[sum, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>:​House:​col:​Properties:​property_type_code] <br>5. COMPARATIVE[#1, #3, comparative:​>:​1:​col:​Properties:​room_count] <br>6. UNION[#4, #5] <br>7. PROJECT[col:​Properties:​property_name, #6] <br> | - | hard | 
 ***
 Exec acc: **0.7500**
