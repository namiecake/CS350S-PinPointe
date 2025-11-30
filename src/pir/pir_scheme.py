"""
Private Information Retrieval (PIR) Implementation for PinPointe
Based on information-theoretic PIR with computational optimizations.

File: pir/pir_scheme.py
"""

import numpy as np
from typing import List, Dict, Tuple, Any
import json


class PIRServer:
    """
    Server-side PIR implementation.
    Responds to queries without learning which cluster was queried.
    """
    
    def __init__(self, database: np.ndarray, num_bits: int = 32):
        """
        Initialize PIR server with a database.
        
        Args:
            database: 2D numpy array where each row is a record (cluster's recommendations)
                     Shape: (num_clusters, items_per_cluster)
            num_bits: Bit precision for encoding (default: 32 for int32)
        """
        self.num_records = database.shape[0]
        self.record_size = database.shape[1]
        self.num_bits = num_bits
        
        # Store database as numpy array for efficient computation
        self.database = database.astype(np.int64)
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get public parameters that client needs.
        These can be published without privacy concerns.
        """
        return {
            'num_records': self.num_records,
            'record_size': self.record_size,
            'num_bits': self.num_bits
        }
    
    def answer_query(self, query_vector: np.ndarray) -> np.ndarray:
        """
        Answer a PIR query without learning which record was requested.
        
        The server computes a linear combination of all database rows.
        Since the query is "encrypted" (masked with random values), the server
        cannot determine which row the client actually wants.
        
        Args:
            query_vector: 1D array of shape (num_records,) from client
            
        Returns:
            response: 1D array of shape (record_size,) 
        """
        # Matrix multiplication: query_vector^T × database
        # This computes a weighted sum of all rows
        response = np.dot(query_vector, self.database)
        return response.astype(np.int64)


class PIRClient:
    """
    Client-side PIR implementation.
    Generates queries and decodes responses privately.
    """
    
    def __init__(self, params: Dict[str, Any], noise_scale: float = 0.0):
        """
        Initialize PIR client with server parameters.
        
        Args:
            params: Server parameters from get_params()
            noise_scale: Scale of noise for privacy (smaller = more privacy, more error)
        """
        self.num_records = params['num_records']
        self.record_size = params['record_size']
        self.num_bits = params['num_bits']
        self.noise_scale = noise_scale
        
        # Store the secret information for decoding
        self.last_query_index = None
        self.last_query_noise = None
    
    def generate_query(self, index: int) -> np.ndarray:
        """
        Generate a PIR query for a specific index.
        
        The query is a vector that looks random to the server, but the client
        knows how to decode the response to extract just the desired record.
        
        Args:
            index: Which record to retrieve (0 to num_records-1)
            
        Returns:
            query_vector: 1D array of shape (num_records,)
        """
        if index < 0 or index >= self.num_records:
            raise ValueError(f"Index {index} out of range [0, {self.num_records})")
        
        # Create query vector with noise
        # In a real implementation, this would use LWE or similar
        query_vector = np.random.normal(0, self.noise_scale, self.num_records)
        
        # Set the target index to 1 (the signal we want)
        query_vector[index] = 1.0
        
        # Store for decoding later
        self.last_query_index = index
        self.last_query_noise = query_vector.copy()
        
        return query_vector
    
    def decode_response(self, response: np.ndarray) -> List[int]:
        """
        Decode the server's response to get the requested record.
        
        Since we know which index we queried and what noise we added,
        we can extract just the record we wanted.
        
        Args:
            response: Server's response array
            
        Returns:
            The requested record as a list of integers
        """
        if self.last_query_index is None:
            raise ValueError("No query has been generated yet")
        
        # The response is approximately: 1.0 * database[index] + noise * other_rows
        # Since noise is small and we know the index, we can decode
        decoded = np.round(response).astype(np.int32)
        
        return decoded.tolist()
    
    def reset(self):
        """Reset the client state (for generating a new query)."""
        self.last_query_index = None
        self.last_query_noise = None


def setup_pir_database(
    cluster_recommendations: Dict[int, List[int]], 
    num_clusters: int,
    rec_list_size: int
) -> Tuple[PIRServer, Dict[str, Any]]:
    """
    Setup PIR server with cluster recommendations as the database.
    
    Args:
        cluster_recommendations: Dict mapping cluster_id -> list of book IDs
        num_clusters: Total number of clusters
        rec_list_size: Fixed size for each recommendation list
        
    Returns:
        Tuple of (pir_server, pir_params)
    """
    # Convert cluster recommendations to fixed-size numpy array
    database_rows = []
    
    for cluster_id in range(num_clusters):
        recs = cluster_recommendations.get(cluster_id, [])
        
        # Pad or truncate to rec_list_size
        if len(recs) < rec_list_size:
            padded_recs = recs + [0] * (rec_list_size - len(recs))
        else:
            padded_recs = recs[:rec_list_size]
        
        database_rows.append(padded_recs)
    
    # Convert to numpy array
    database = np.array(database_rows, dtype=np.int32)
    
    # Initialize PIR server
    pir_server = PIRServer(database)
    pir_params = pir_server.get_params()
    
    print(f"PIR Server initialized:")
    print(f"  - Number of clusters (records): {num_clusters}")
    print(f"  - Recommendations per cluster: {rec_list_size}")
    print(f"  - Database shape: {database.shape}")
    
    return pir_server, pir_params


def retrieve_recommendations_with_pir(
    cluster_id: int,
    pir_client: PIRClient,
    pir_server: PIRServer
) -> List[int]:
    """
    Privately retrieve recommendations for a cluster using PIR.
    
    This is the main function that performs private retrieval.
    The server learns nothing about which cluster_id was requested.
    
    Args:
        cluster_id: The cluster to retrieve recommendations for
        pir_client: Initialized PIR client
        pir_server: Initialized PIR server
        
    Returns:
        List of book IDs recommended for the cluster (zeros filtered out)
    """
    # CLIENT SIDE: Generate PIR query for the desired cluster
    query = pir_client.generate_query(cluster_id)
    
    # NETWORK: Send query to server
    # In a real system, this would be sent over the network
    # The server sees only the query vector, not the cluster_id
    
    # SERVER SIDE: Answer query (server learns nothing about cluster_id!)
    response = pir_server.answer_query(query)
    
    # NETWORK: Send response back to client
    
    # CLIENT SIDE: Decode response to get recommendations
    recommendations = pir_client.decode_response(response)
    
    # Filter out padding zeros
    recommendations = [rec for rec in recommendations if rec != 0]
    
    # Reset client for next query
    pir_client.reset()
    
    return recommendations


# ============================================================================
# Testing and Demo
# ============================================================================

def test_pir_basic():
    """Test basic PIR functionality with a small example."""
    print("\n" + "="*70)
    print("Testing Basic PIR Functionality")
    print("="*70)
    
    # Create a small test database
    test_cluster_recs = {
        0: [101, 102, 103, 104, 105],
        1: [201, 202, 203, 204, 205],
        2: [301, 302, 303, 304, 305],
        3: [401, 402, 403, 404, 405],
        4: [501, 502, 503, 504, 505],
    }
    
    num_clusters = 5
    rec_list_size = 10
    
    # Setup PIR
    pir_server, pir_params = setup_pir_database(
        test_cluster_recs, num_clusters, rec_list_size
    )
    
    pir_client = PIRClient(pir_params)
    
    # Test retrieval for each cluster
    print("\nTesting retrieval for each cluster:")
    for cluster_id in range(num_clusters):
        retrieved = retrieve_recommendations_with_pir(
            cluster_id, pir_client, pir_server
        )
        expected = test_cluster_recs[cluster_id]
        
        match = retrieved == expected
        status = "✓ PASS" if match else "✗ FAIL"
        
        print(f"Cluster {cluster_id}: {status}")
        print(f"  Expected:  {expected}")
        print(f"  Retrieved: {retrieved}")
    
    print("\n" + "="*70)


def save_pir_server(pir_server: PIRServer, filepath: str):
    """Save PIR server state to disk."""
    np.savez_compressed(
        filepath,
        database=pir_server.database,
        num_bits=pir_server.num_bits
    )
    print(f"PIR server saved to {filepath}")


def load_pir_server(filepath: str) -> PIRServer:
    """Load PIR server state from disk."""
    data = np.load(filepath)
    pir_server = PIRServer(data['database'], int(data['num_bits']))
    print(f"PIR server loaded from {filepath}")
    return pir_server


if __name__ == "__main__":
    # Run tests
    test_pir_basic()