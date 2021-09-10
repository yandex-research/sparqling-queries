 | Idx | Table      | Column | Primary Key | Foreign Key | 
 | ----------- | ----------- | ----------- | ----------- | ----------- | 
  | 0 |  | * |   |   | 
 | 1 | **TV_Channel** | id | + |   | 
 | 2 |   | series_name |   |   | 
 | 3 |   | Country |   |   | 
 | 4 |   | Language |   |   | 
 | 5 |   | Content |   |   | 
 | 6 |   | Pixel_aspect_ratio_PAR |   |   | 
 | 7 |   | Hight_definition_TV |   |   | 
 | 8 |   | Pay_per_view_PPV |   |   | 
 | 9 |   | Package_Option |   |   | 
 | 10 | **TV_series** | id | + |   | 
 | 11 |   | Episode |   |   | 
 | 12 |   | Air_Date |   |   | 
 | 13 |   | Rating |   |   | 
 | 14 |   | Share |   |   | 
 | 15 |   | 18_49_Rating_Share |   |   | 
 | 16 |   | Viewers_m |   |   | 
 | 17 |   | Weekly_Rank |   |   | 
 | 18 |   | Channel |   | --> 1 | 
 | 19 | **Cartoon** | id | + |   | 
 | 20 |   | Title |   |   | 
 | 21 |   | Directed_by |   |   | 
 | 22 |   | Written_by |   |   | 
 | 23 |   | Original_air_date |   |   | 
 | 24 |   | Production_code |   |   | 
 | 25 |   | Channel |   | --> 1 | 
 
  | Index | Question  | SQL | gold QDMR | pred QDMR | Exec | SQL hardness |
  | ----------- | ----------- | ----------- |  ----------- | ----------- | ----------- | ----------- | 
 | SPIDER_dev_585 | List the title of all cartoons in alphabetical order. | SELECT Title FROM Cartoon ORDER BY title |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Title, #1] <br>3. SORT[#2, #2, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_dev_586 | What are the titles of the cartoons sorted alphabetically? | SELECT Title FROM Cartoon ORDER BY title |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Title, #1] <br>3. SORT[#2, #2, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_dev_587 | List all cartoon directed by "Ben Jones". | SELECT Title FROM Cartoon WHERE Directed_by = "Ben Jones"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Directed_by, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br> | - | easy | 
  | SPIDER_dev_588 | What are the names of all cartoons directed by Ben Jones? | SELECT Title FROM Cartoon WHERE Directed_by = "Ben Jones"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Directed_by, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>4. PROJECT[col:​Cartoon:​Title, #3] <br> | + | easy | 
  | SPIDER_dev_589 | How many cartoons were written by "Joseph Kuhr"? | SELECT count(*) FROM Cartoon WHERE Written_by = "Joseph Kuhr"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Written_by, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Joseph Kuhr:​col:​Cartoon:​Written_by] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_dev_590 | What is the number of cartoones written by Joseph Kuhr? | SELECT count(*) FROM Cartoon WHERE Written_by = "Joseph Kuhr"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Joseph Kuhr:​col:​Cartoon:​Written_by] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_dev_591 | list all cartoon titles and their directors ordered by their air date | SELECT title ,  Directed_by FROM Cartoon ORDER BY Original_air_date |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Title, #1] <br>3. PROJECT[col:​Cartoon:​Directed_by, #1] <br>4. PROJECT[col:​Cartoon:​Directed_by, #1] <br>5. UNION[#2, #3] <br>6. SORT[#5, #4, sortdir:​ascending] <br> | - | medium | 
  | SPIDER_dev_592 | What is the name and directors of all the cartoons that are ordered by air date? | SELECT title ,  Directed_by FROM Cartoon ORDER BY Original_air_date |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Title, #1] <br>3. PROJECT[col:​Cartoon:​Directed_by, #1] <br>4. PROJECT[col:​Cartoon:​Original_air_date, #1] <br>5. UNION[#2, #3] <br>6. SORT[#5, #4, sortdir:​ascending] <br> | + | medium | 
  | SPIDER_dev_593 | List the title of all cartoon directed by "Ben Jones" or "Brandon Vietti". | SELECT Title FROM Cartoon WHERE Directed_by = "Ben Jones" OR Directed_by = "Brandon Vietti"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Directed_by, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Brandon Vietti:​col:​Cartoon:​Directed_by] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​Cartoon:​Title, #5] <br> | + | medium | 
  | SPIDER_dev_594 | What are the titles of all cartoons directed by Ben Jones or Brandon Vietti? | SELECT Title FROM Cartoon WHERE Directed_by = "Ben Jones" OR Directed_by = "Brandon Vietti"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Directed_by, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Brandon Vietti:​col:​Cartoon:​Directed_by] <br>5. UNION[#3, #4] <br>6. PROJECT[col:​Cartoon:​Title, #5] <br> | + | medium | 
  | SPIDER_dev_595 | Which country has the most of TV Channels? List the country and number of TV Channels it has. | SELECT Country ,  count(*) FROM TV_Channel GROUP BY Country ORDER BY count(*) DESC LIMIT 1; |  | 1. SELECT[col:​TV_Channel:​Country] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. AGGREGATE[max, #3] <br>5. COMPARATIVE[#1, #3, comparative:​=:​English:​col:​TV_Channel:​Language] <br>6. UNION[#4, #5] <br> | - | hard | 
  | SPIDER_dev_596 | What is the country with the most number of TV Channels and how many does it have? | SELECT Country ,  count(*) FROM TV_Channel GROUP BY Country ORDER BY count(*) DESC LIMIT 1; |  | 1. SELECT[col:​TV_Channel:​Country] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[tbl:​TV_Channel, #4] <br>6. AGGREGATE[count, #5] <br>7. UNION[#4, #6] <br> | + | hard | 
  | SPIDER_dev_597 | List the number of different series names and contents in the TV Channel table. | SELECT count(DISTINCT series_name) ,  count(DISTINCT content) FROM TV_Channel; |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​series_name, #1] <br>3.*(distinct)* PROJECT[None, #2] <br>4. UNION[#3, #2] <br> | - | medium | 
  | SPIDER_dev_598 | How many different series and contents are listed in the TV Channel table? | SELECT count(DISTINCT series_name) ,  count(DISTINCT content) FROM TV_Channel; |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​series_name, #1] <br>3.*(distinct)* PROJECT[None, #2] <br>4. AGGREGATE[count, #3] <br>5. UNION[#2, #4] <br> | - | medium | 
  | SPIDER_dev_599 | What is the content of TV Channel with serial name "Sky Radio"? | SELECT Content FROM TV_Channel WHERE series_name = "Sky Radio"; |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​series_name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Sky Radio:​col:​TV_Channel:​series_name] <br>4. PROJECT[col:​TV_Channel:​Content, #3] <br> | + | easy | 
  | SPIDER_dev_600 | What is the content of the series Sky Radio? | SELECT Content FROM TV_Channel WHERE series_name = "Sky Radio"; |  | 1. SELECT[val:​TV_Channel:​series_name:​Sky Radio] <br>2. PROJECT[col:​TV_Channel:​Content, #1] <br> | + | easy | 
  | SPIDER_dev_601 | What is the Package Option of TV Channel with serial name "Sky Radio"? | SELECT Package_Option FROM TV_Channel WHERE series_name = "Sky Radio"; |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​series_name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Sky Radio:​col:​TV_Channel:​series_name] <br>4. PROJECT[col:​TV_Channel:​Package_Option, #3] <br> | + | easy | 
  | SPIDER_dev_602 | What are the Package Options of the TV Channels whose series names are Sky Radio? | SELECT Package_Option FROM TV_Channel WHERE series_name = "Sky Radio"; |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​series_name, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Sky Radio:​col:​TV_Channel:​series_name] <br>4. PROJECT[col:​TV_Channel:​Package_Option, #3] <br> | + | easy | 
  | SPIDER_dev_603 | How many TV Channel using language English? | SELECT count(*) FROM TV_Channel WHERE LANGUAGE = "English"; |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​Language, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​English:​col:​TV_Channel:​Language] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_dev_604 | How many TV Channels use the English language? | SELECT count(*) FROM TV_Channel WHERE LANGUAGE = "English"; |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​Language, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​English:​col:​TV_Channel:​Language] <br>4. AGGREGATE[count, #3] <br> | + | easy | 
  | SPIDER_dev_605 | List the language used least number of TV Channel. List language and number of TV Channel. | SELECT LANGUAGE ,  count(*) FROM TV_Channel GROUP BY LANGUAGE ORDER BY count(*) ASC LIMIT 1; |  | 1. SELECT[col:​TV_Channel:​Language] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​min:​None, #1, #3] <br>5. PROJECT[tbl:​TV_Channel, #4] <br>6. AGGREGATE[count, #5] <br>7. UNION[#4, #6] <br> | + | hard | 
  | SPIDER_dev_606 | What are the languages used by the least number of TV Channels and how many channels use it? | SELECT LANGUAGE ,  count(*) FROM TV_Channel GROUP BY LANGUAGE ORDER BY count(*) ASC LIMIT 1; |  | 1. SELECT[col:​TV_Channel:​Language] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. SUPERLATIVE[comparative:​min:​None, #1, #3] <br>5. PROJECT[tbl:​TV_Channel, #4] <br>6. GROUP[count, #5, #4] <br>7. UNION[#4, #6] <br> | + | hard | 
  | SPIDER_dev_607 | List each language and the number of TV Channels using it. | SELECT LANGUAGE ,  count(*) FROM TV_Channel GROUP BY LANGUAGE |  | 1. SELECT[col:​TV_Channel:​Language] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_dev_608 | For each language, list the number of TV Channels that use it. | SELECT LANGUAGE ,  count(*) FROM TV_Channel GROUP BY LANGUAGE |  | 1. SELECT[col:​TV_Channel:​Language] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_dev_609 | What is the TV Channel that shows the cartoon "The Rise of the Blue Beetle!"? List the TV Channel's series name. | SELECT T1.series_name FROM TV_Channel AS T1 JOIN Cartoon AS T2 ON T1.id = T2.Channel WHERE T2.Title = "The Rise of the Blue Beetle!"; |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, tbl:​Cartoon] <br>3. PROJECT[col:​TV_Channel:​series_name, #2] <br> | - | medium | 
  | SPIDER_dev_610 | What is the series name of the TV Channel that shows the cartoon "The Rise of the Blue Beetle"? | SELECT T1.series_name FROM TV_Channel AS T1 JOIN Cartoon AS T2 ON T1.id = T2.Channel WHERE T2.Title = "The Rise of the Blue Beetle!"; |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, tbl:​Cartoon] <br>3. PROJECT[col:​TV_Channel:​series_name, #2] <br> | - | medium | 
  | SPIDER_dev_611 | List the title of all  Cartoons showed on TV Channel with series name "Sky Radio". | SELECT T2.Title FROM TV_Channel AS T1 JOIN Cartoon AS T2 ON T1.id = T2.Channel WHERE T1.series_name = "Sky Radio"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. PROJECT[col:​TV_Channel:​series_name, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​Sky Radio:​col:​TV_Channel:​series_name] <br>5. PROJECT[col:​Cartoon:​Title, #4] <br> | + | medium | 
  | SPIDER_dev_612 | What is the title of all the cartools that are on the TV Channel with the series name "Sky Radio"? | SELECT T2.Title FROM TV_Channel AS T1 JOIN Cartoon AS T2 ON T1.id = T2.Channel WHERE T1.series_name = "Sky Radio"; |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Sky Radio:​col:​TV_Channel:​series_name] <br>4. PROJECT[col:​Cartoon:​Title, #3] <br> | + | medium | 
  | SPIDER_dev_613 | List the Episode of all TV series sorted by rating. | SELECT Episode FROM TV_series ORDER BY rating |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Episode, #1] <br>3. PROJECT[col:​TV_series:​Rating, #1] <br>4. SORT[#2, #3, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_dev_614 | What are all of the episodes ordered by ratings? | SELECT Episode FROM TV_series ORDER BY rating |  | 1. SELECT[col:​TV_series:​Episode] <br>2. PROJECT[col:​TV_series:​Rating, #1] <br>3. SORT[#1, #2, sortdir:​ascending] <br> | + | easy | 
  | SPIDER_dev_615 | List top 3 highest Rating  TV series. List the TV series's Episode and Rating. | SELECT Episode ,  Rating FROM TV_series ORDER BY Rating DESC LIMIT 3; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Rating, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​3.0:​col:​TV_series:​Rating] <br>4. SUPERLATIVE[comparative:​max:​None, #1, #3] <br>5. PROJECT[col:​TV_series:​Episode, #4] <br>6. PROJECT[col:​TV_series:​Rating, #4] <br>7. UNION[#5, #6] <br> | - | medium | 
  | SPIDER_dev_616 | What are 3 most highly rated episodes in the TV series table and what were those ratings? | SELECT Episode ,  Rating FROM TV_series ORDER BY Rating DESC LIMIT 3; |  | 1. SELECT[col:​TV_series:​Episode] <br>2. PROJECT[col:​TV_series:​Rating, #1] <br>3. COMPARATIVE[#2, #2, comparative:​=:​3.0:​col:​TV_series:​Rating] <br>4. COMPARATIVE[#2, #2, comparative:​=:​3.0:​col:​TV_series:​Rating] <br>5. UNION[#3, #4] <br> | - | medium | 
  | SPIDER_dev_617 | What is minimum and maximum share of TV series? | SELECT max(SHARE) , min(SHARE) FROM TV_series; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Share, #1] <br>3. AGGREGATE[min, #2] <br>4. AGGREGATE[max, #2] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_dev_618 | What is the maximum and minimum share for the TV series? | SELECT max(SHARE) , min(SHARE) FROM TV_series; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Share, #1] <br>3. AGGREGATE[max, #2] <br>4. AGGREGATE[min, #2] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_dev_619 | What is the air date of TV series with Episode "A Love of a Lifetime"? | SELECT Air_Date FROM TV_series WHERE Episode = "A Love of a Lifetime"; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Episode, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​A Love of a Lifetime:​col:​TV_series:​Episode] <br>4. PROJECT[col:​TV_series:​Air_Date, #3] <br> | + | easy | 
  | SPIDER_dev_620 | When did the episode "A Love of a Lifetime" air? | SELECT Air_Date FROM TV_series WHERE Episode = "A Love of a Lifetime"; |  | 1. SELECT[col:​TV_series:​Episode] <br>2. PROJECT[col:​TV_series:​Air_Date, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​A Love of a Lifetime:​col:​TV_series:​Episode] <br> | + | easy | 
  | SPIDER_dev_621 | What is Weekly Rank of TV series with Episode "A Love of a Lifetime"? | SELECT Weekly_Rank FROM TV_series WHERE Episode = "A Love of a Lifetime"; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Episode, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​A Love of a Lifetime:​col:​TV_series:​Episode] <br>4. PROJECT[col:​TV_series:​Weekly_Rank, #3] <br> | + | easy | 
  | SPIDER_dev_622 | What is the weekly rank for the episode "A Love of a Lifetime"? | SELECT Weekly_Rank FROM TV_series WHERE Episode = "A Love of a Lifetime"; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Weekly_Rank, #1] <br>3. COMPARATIVE[#2, #1, comparative:​=:​A Love of a Lifetime:​col:​TV_series:​Episode] <br> | + | easy | 
  | SPIDER_dev_623 | What is the TV Channel of TV series with Episode "A Love of a Lifetime"? List the TV Channel's series name. | SELECT T1.series_name FROM TV_Channel AS T1 JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T2.Episode = "A Love of a Lifetime"; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Episode, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​A Love of a Lifetime:​col:​TV_series:​Episode] <br>4. PROJECT[col:​TV_Channel:​series_name, #3] <br> | + | medium | 
  | SPIDER_dev_624 | What is the name of the series that has the episode "A Love of a Lifetime"? | SELECT T1.series_name FROM TV_Channel AS T1 JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T2.Episode = "A Love of a Lifetime"; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[col:​TV_series:​Episode, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​A Love of a Lifetime:​col:​TV_series:​Episode] <br>4. PROJECT[col:​TV_Channel:​series_name, #3] <br> | + | medium | 
  | SPIDER_dev_625 | List the Episode of all  TV series showed on TV Channel with series name "Sky Radio". | SELECT T2.Episode FROM TV_Channel AS T1 JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T1.series_name = "Sky Radio"; |  | 1. SELECT[tbl:​TV_series] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Sky Radio:​col:​TV_Channel:​series_name] <br>4. PROJECT[col:​TV_series:​Episode, #3] <br> | + | medium | 
  | SPIDER_dev_626 | What is the episode for the TV series named "Sky Radio"? | SELECT T2.Episode FROM TV_Channel AS T1 JOIN TV_series AS T2 ON T1.id = T2.Channel WHERE T1.series_name = "Sky Radio"; |  | 1. SELECT[tbl:​TV_series] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Sky Radio:​col:​TV_Channel:​series_name] <br>3. PROJECT[col:​TV_series:​Episode, #2] <br> | + | medium | 
  | SPIDER_dev_627 | Find the number of cartoons directed by each of the listed directors. | SELECT count(*) ,  Directed_by FROM cartoon GROUP BY Directed_by |  | 1. SELECT[col:​Cartoon:​Directed_by] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_dev_628 | How many cartoons did each director create? | SELECT count(*) ,  Directed_by FROM cartoon GROUP BY Directed_by |  | 1. SELECT[col:​Cartoon:​Directed_by] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. GROUP[count, #2, #1] <br>4. UNION[#1, #3] <br> | + | medium | 
  | SPIDER_dev_629 | Find the production code and channel of the most recently aired cartoon. | select production_code ,  channel from cartoon order by original_air_date desc limit 1 |  | 1. SELECT[tbl:​Cartoon] <br>2. PROJECT[col:​Cartoon:​Original_air_date, #1] <br>3. SUPERLATIVE[comparative:​max:​None, #1, #2] <br>4. PROJECT[col:​Cartoon:​Production_code, #3] <br>5. PROJECT[col:​Cartoon:​Channel, #3] <br>6. UNION[#4, #5] <br> | + | medium | 
  | SPIDER_dev_630 | What is the produdction code and channel of the most recent cartoon? | select production_code ,  channel from cartoon order by original_air_date desc limit 1 |  | 1. SELECT[tbl:​Cartoon] <br>2. SUPERLATIVE[comparative:​max:​None, #1, #1] <br>3. PROJECT[col:​Cartoon:​Production_code, #2] <br>4. PROJECT[col:​Cartoon:​Channel, #2] <br>5. UNION[#3, #4] <br> | - | medium | 
  | SPIDER_dev_631 | Find the package choice and series name of the TV channel that has high definition TV. | SELECT package_option ,  series_name FROM TV_Channel WHERE hight_definition_TV  =  "yes" |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Option:​col:​TV_Channel:​Package_Option] <br>3. PROJECT[col:​TV_Channel:​Package_Option, #2] <br>4. PROJECT[col:​TV_Channel:​series_name, #2] <br>5. UNION[#3, #4] <br> | - | medium | 
  | SPIDER_dev_632 | What are the package options and the name of the series for the TV Channel that supports high definition TV? | SELECT package_option ,  series_name FROM TV_Channel WHERE hight_definition_TV  =  "yes" |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[col:​TV_Channel:​Hight_definition_TV, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Option:​col:​TV_Channel:​Package_Option] <br>4. PROJECT[col:​TV_Channel:​Package_Option, #3] <br>5. PROJECT[col:​TV_Channel:​series_name, #3] <br>6. UNION[#4, #5] <br> | - | medium | 
  | SPIDER_dev_633 | which countries' tv channels are playing some cartoon written by Todd Casey? | SELECT T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.written_by  =  'Todd Casey' |  | 1. SELECT[col:​TV_Channel:​Country] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Todd Casey:​col:​Cartoon:​Written_by] <br>4. PROJECT[col:​TV_Channel:​Country, #3] <br> | + | medium | 
  | SPIDER_dev_634 | What are the countries that have cartoons on TV that were written by Todd Casey? | SELECT T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.written_by  =  'Todd Casey' |  | 1. SELECT[col:​TV_Channel:​Country] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. PROJECT[col:​Cartoon:​Written_by, #2] <br>4. COMPARATIVE[#1, #3, comparative:​=:​Todd Casey:​col:​Cartoon:​Written_by] <br> | + | medium | 
  | SPIDER_dev_635 | which countries' tv channels are not playing any cartoon written by Todd Casey? | SELECT country FROM TV_Channel EXCEPT SELECT T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.written_by  =  'Todd Casey' |  | 1. SELECT[col:​TV_Channel:​Country] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Todd Casey:​col:​Cartoon:​Written_by] <br>4. DISCARD[#1, #3] <br> | + | hard | 
  | SPIDER_dev_636 | What are the countries that are not playing cartoons written by Todd Casey? | SELECT country FROM TV_Channel EXCEPT SELECT T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.written_by  =  'Todd Casey' |  | 1. SELECT[col:​TV_Channel:​Country] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Todd Casey:​col:​Cartoon:​Written_by] <br>4. DISCARD[#1, #3] <br> | + | hard | 
  | SPIDER_dev_637 | Find the series name and country of the tv channel that is playing some cartoons directed by Ben Jones and Michael Chang? | SELECT T1.series_name ,  T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.directed_by  =  'Michael Chang' INTERSECT SELECT T1.series_name ,  T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.directed_by  =  'Ben Jones' |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Michael Chang:​col:​Cartoon:​Directed_by] <br>5. INTERSECTION[#1, #3, #4] <br>6. PROJECT[col:​TV_Channel:​series_name, #5] <br>7. PROJECT[col:​TV_Channel:​Country, #5] <br>8. UNION[#6, #7] <br> | + | extra | 
  | SPIDER_dev_638 | What is the series name and country of all TV channels that are playing cartoons directed by Ben Jones and cartoons directed by Michael Chang? | SELECT T1.series_name ,  T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.directed_by  =  'Michael Chang' INTERSECT SELECT T1.series_name ,  T1.country FROM TV_Channel AS T1 JOIN cartoon AS T2 ON T1.id = T2.Channel WHERE T2.directed_by  =  'Ben Jones' |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>4. COMPARATIVE[#1, #2, comparative:​=:​Michael Chang:​col:​Cartoon:​Directed_by] <br>5. INTERSECTION[#1, #3, #4] <br>6. PROJECT[col:​TV_Channel:​series_name, #5] <br>7. PROJECT[col:​TV_Channel:​Country, #5] <br>8. UNION[#6, #7] <br> | + | extra | 
  | SPIDER_dev_639 | find the pixel aspect ratio and nation of the tv channels that do not use English. | SELECT Pixel_aspect_ratio_PAR ,  country FROM tv_channel WHERE LANGUAGE != 'English' |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, comparative:​!=:​English:​col:​TV_Channel:​Language] <br>3. PROJECT[col:​TV_Channel:​Pixel_aspect_ratio_PAR, #2] <br>4. PROJECT[col:​TV_Channel:​Country, #2] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_dev_640 | What is the pixel aspect ratio and country of origin for all TV channels that do not use English? | SELECT Pixel_aspect_ratio_PAR ,  country FROM tv_channel WHERE LANGUAGE != 'English' |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, comparative:​!=:​English:​col:​TV_Channel:​Language] <br>3. PROJECT[col:​TV_Channel:​Pixel_aspect_ratio_PAR, #2] <br>4. PROJECT[col:​TV_Channel:​Country, #2] <br>5. UNION[#3, #4] <br> | + | medium | 
  | SPIDER_dev_641 | find id of the tv channels that from the countries where have more than two tv channels. | SELECT id FROM tv_channel GROUP BY country HAVING count(*)  >  2 |  | 1. SELECT[col:​TV_Channel:​Country] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>:​2] <br>5. PROJECT[col:​TV_Channel:​Country, #4] <br> | - | easy | 
  | SPIDER_dev_642 | What are the ids of all tv channels that have more than 2 TV channels? | SELECT id FROM tv_channel GROUP BY country HAVING count(*)  >  2 |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[tbl:​TV_Channel, #1] <br>3. GROUP[count, #2, #1] <br>4. COMPARATIVE[#1, #3, comparative:​>:​2] <br>5. PROJECT[col:​TV_Channel:​id, #4] <br> | - | easy | 
  | SPIDER_dev_643 | find the id of tv channels that do not play any cartoon directed by Ben Jones. | SELECT id FROM TV_Channel EXCEPT SELECT channel FROM cartoon WHERE directed_by  =  'Ben Jones' |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​Cartoon:​id, #3] <br> | - | hard | 
  | SPIDER_dev_644 | What are the ids of the TV channels that do not have any cartoons directed by Ben Jones? | SELECT id FROM TV_Channel EXCEPT SELECT channel FROM cartoon WHERE directed_by  =  'Ben Jones' |  | 1. SELECT[tbl:​TV_Channel] <br>2. PROJECT[tbl:​Cartoon, #1] <br>3. COMPARATIVE[#1, #2, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>4. DISCARD[#1, #3] <br>5. PROJECT[col:​Cartoon:​id, #4] <br> | - | hard | 
  | SPIDER_dev_645 | find the package option of the tv channel that do not have any cartoon directed by Ben Jones. | SELECT package_option FROM TV_Channel WHERE id NOT IN (SELECT channel FROM cartoon WHERE directed_by  =  'Ben Jones') |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​TV_Channel:​Package_Option, #3] <br> | + | hard | 
  | SPIDER_dev_646 | What are the package options of all tv channels that are not playing any cartoons directed by Ben Jones? | SELECT package_option FROM TV_Channel WHERE id NOT IN (SELECT channel FROM cartoon WHERE directed_by  =  'Ben Jones') |  | 1. SELECT[tbl:​TV_Channel] <br>2. COMPARATIVE[#1, #1, comparative:​=:​Ben Jones:​col:​Cartoon:​Directed_by] <br>3. DISCARD[#1, #2] <br>4. PROJECT[col:​TV_Channel:​Package_Option, #3] <br> | + | hard | 
 ***
 Exec acc: **0.7419**