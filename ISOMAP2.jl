include("misc.jl")
include("PCA.jl")
include("findMin.jl")

function MDS(X)
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))

    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2

            # Gradient
            df = s
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
        end
    end
    return (f,G[:])
end

function ISOMAP(X)
	(n,d) = size(X)

	# Find the neighbours of each point by KNN
	model = knn(X,X[:,1],2)
	G = model.predict(X)

	# Compute the edge weights, where the weights are the distance between neighbours
	weights = zeros(n,n) 
    for i in 1:n
        for j in 1:n
            weights[i,j] = min(G[i,j],G[j,i])
        end
    end

	# Compute weighted shortest path between all points using Dijkstra
	D = zeros(n,n)

    maxdist = 0
	for i in 1:n
		for j in 1:n
            D[i,j] = dijkstra(weights, i, j)
            if D[i,j] > maxdist && D[i,j] < Inf
                maxdist = D[i,j]
            end
		end
	end
    
    for i in 1:n
        for j in 1:n
            D[i,j] = min(D[i,j], maxdist)
        end
    end

    #print(D)	
	
    # Run MDS on the distances
	Z = MDS(D)

	 # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(z) = stress(z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z


end

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin

  yhat = zeros(t)
  G = zeros(t,n)
  fill!(G, Inf)

  for i=1:t
    distances = zeros(n)
    for j=1:n
        # The 2-norm (Euclidean distance) is the default norm in Julia.
	    if i == j
		    distances[j] = Inf
	    else
            distances[j] = norm(Xhat[i,:] - X[j,:], 2)
	    end
    end

    # Assign the most common label among k nearest training examples.
    nearest_k_indices = sortperm(distances)[1:k]
    for m=1:k
	    ind = nearest_k_indices[m]
	    G[i,ind] = distances[ind]
    end
    # yhat[i] = mode(y[nearest_k_indices])
  end

  return G
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end
