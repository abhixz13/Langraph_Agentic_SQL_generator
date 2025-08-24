#!/usr/bin/env python3
"""
Test the expected SQL query manually
"""

import sqlite3

def test_expected_sql():
    """Test the expected SQL query for the user query"""
    
    user_query = "Show me the top 2 customers of Intersight SaaS for each year"
    
    print("=" * 80)
    print("EXPECTED SQL QUERY")
    print("=" * 80)
    print(f"User Query: {user_query}")
    print()
    
    # Expected SQL query
    expected_sql = """
    WITH RankedCustomers AS (
        SELECT 
            CUSTOMER_NAME,
            YEAR,
            ACTUAL_BOOKINGS,
            ROW_NUMBER() OVER (
                PARTITION BY YEAR 
                ORDER BY ACTUAL_BOOKINGS DESC
            ) as rank
        FROM raw_data 
        WHERE IntersightConsumption = 'SaaS'
    )
    SELECT 
        CUSTOMER_NAME,
        YEAR,
        ACTUAL_BOOKINGS
    FROM RankedCustomers 
    WHERE rank <= 2
    ORDER BY YEAR, rank
    """
    
    print("Expected SQL Query:")
    print("-" * 80)
    print(expected_sql.strip())
    print()
    
    # Test the expected SQL
    print("Testing Expected SQL Query:")
    print("-" * 80)
    
    try:
        conn = sqlite3.connect('data/raw_data.db')
        cursor = conn.cursor()
        
        # Execute the expected SQL
        cursor.execute(expected_sql.strip())
        results = cursor.fetchall()
        
        if results:
            print(f"Query returned {len(results)} results:")
            print()
            print("Top 2 SaaS customers by revenue for each year:")
            print("-" * 50)
            
            current_year = None
            for row in results:
                customer_name, year, revenue = row
                if year != current_year:
                    print(f"\nYear {year}:")
                    current_year = year
                print(f"  {customer_name}: ${revenue:,.2f}")
        else:
            print("No results found")
            
        conn.close()
        
    except sqlite3.Error as e:
        print(f"SQL Error: {e}")
        
        # Try a simpler approach if window functions don't work
        print("\nTrying simpler approach without window functions:")
        simple_sql = """
        SELECT 
            CUSTOMER_NAME,
            YEAR,
            ACTUAL_BOOKINGS
        FROM raw_data 
        WHERE IntersightConsumption = 'SaaS'
        ORDER BY YEAR, ACTUAL_BOOKINGS DESC
        LIMIT 20
        """
        
        try:
            cursor = conn.cursor()
            cursor.execute(simple_sql.strip())
            results = cursor.fetchall()
            
            if results:
                print(f"Simple query returned {len(results)} results:")
                for row in results[:10]:  # Show first 10
                    customer_name, year, revenue = row
                    print(f"  {customer_name} - {year}: ${revenue:,.2f}")
            else:
                print("No results found")
                
        except sqlite3.Error as e2:
            print(f"Simple SQL Error: {e2}")
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("AI Generated SQL:")
    print("SELECT CUSTOMER_NAME, YEAR, IntersightConsumption FROM raw_data WHERE YEAR >= 2021")
    print()
    print("Expected SQL (with window functions):")
    print(expected_sql.strip())
    print()
    print("The AI generated SQL is missing:")
    print("- Filtering for SaaS customers (IntersightConsumption = 'SaaS')")
    print("- Window functions to get top 2 per year")
    print("- Revenue ranking (ACTUAL_BOOKINGS)")
    print("- Proper grouping by year")

if __name__ == "__main__":
    test_expected_sql()
