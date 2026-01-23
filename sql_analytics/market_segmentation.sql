-- Market analysis of property types by price -- 
SELECT title, price, property_type, 
	NTILE(4) OVER (order by price ASC) as price_quartile, 
	case
		when NTILE(4) over (order by price ASC) = 1 then 'Budget'
		when NTILE(4) over (order by price ASC) = 2 then 'Mid'
		when NTILE(4) over (order by price ASC) = 3 then 'Premium'
		else 'Luxury'
	end as market_segment
from listings
order by price asc;

-- Analysis of price per bed per property type -- 
CREATE VIEW best_value_rentals AS
SELECT 
    title,
    price,
    beds,
    property_type,
    ROUND(price::numeric / NULLIF(beds, 0), 2) as price_per_bed
FROM listings
WHERE beds > 0
AND (price::numeric / NULLIF(beds, 0)) < 800
ORDER BY price_per_bed ASC;