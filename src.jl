using ModelingToolkit
using NeuralPDE
using Flux
using DiffEqFlux
using GalacticOptim

function fitPDESystem!(K, α, µ, ρ₀, ρ₀ˢ, ρ₀ˡ,
                       F₁, F₂,
                       tspan, x₁span, x₂span,
                       Δt, Δx₁, Δx₂;
                       maxiters=1000, strategy = GridTraining())

   @parameters t, x₁, x₂, θ
   @variables u₁(..), u₂(..), v₁(..), v₂(..), σ₁₁(..), σ₁₂(..), σ₂₂(..), p(..)
   @derivatives Dt'~t
   @derivatives Dx₁'~x₁
   @derivatives Dx₂'~x₂

   # System of PDEs
   equations = [
      Dt(u₁(t,x₁,x₂,θ)) + ( Dx₁(σ₁₁(t,x₁,x₂,θ)) + Dx₂(σ₁₂(t,x₁,x₂,θ)) )/ρ₀ˢ + Dx₁(p(t,x₁,x₂,θ))/ρ₀ - F₁(t,x₁,x₂) ~ 0,
      Dt(u₂(t,x₁,x₂,θ)) + ( Dx₁(σ₁₂(t,x₁,x₂,θ)) + Dx₂(σ₂₂(t,x₁,x₂,θ)) )/ρ₀ˢ + Dx₂(p(t,x₁,x₂,θ))/ρ₀ - F₂(t,x₁,x₂) ~ 0,
      Dt(v₁(t,x₁,x₂,θ)) + Dx₁(p(t,x₁,x₂,θ))/ρ₀ - F₁(t,x₁,x₂) ~ 0,
      Dt(v₂(t,x₁,x₂,θ)) + Dx₂(p(t,x₁,x₂,θ))/ρ₀ - F₂(t,x₁,x₂) ~ 0,
      Dt(σ₁₁(t,x₁,x₂,θ)) + (ρ₀ˡ/ρ₀ * K + 4µ/3)Dx₁(u₁(t,x₁,x₂,θ)) + (ρ₀ˡ/ρ₀ * K - 2µ/3)Dx₂(u₂(t,x₁,x₂,θ)) - ρ₀ˢ/ρ₀ * K * ( Dx₁(v₁(t,x₁,x₂,θ)) + Dx₂(v₂(t,x₁,x₂,θ)) ) ~ 0,
      Dt(σ₁₂(t,x₁,x₂,θ)) + µ * ( Dx₂(u₁(t,x₁,x₂,θ)) + Dx₁(u₂(t,x₁,x₂,θ)) ) ~ 0,
      Dt(σ₂₂(t,x₁,x₂,θ)) + (ρ₀ˡ/ρ₀ * K - 2µ/3)Dx₁(u₁(t,x₁,x₂,θ)) + (ρ₀ˡ/ρ₀ * K + 4µ/3)Dx₂(u₂(t,x₁,x₂,θ)) - ρ₀ˢ/ρ₀ * K * ( Dx₁(v₁(t,x₁,x₂,θ)) + Dx₂(v₂(t,x₁,x₂,θ)) ) ~ 0,
      Dt(p(t,x₁,x₂,θ)) - (K - α*ρ₀*ρ₀ˢ)*( Dx₁(u₁(t,x₁,x₂,θ)) + Dx₂(u₂(t,x₁,x₂,θ)) ) + α*ρ₀*ρ₀ˡ*( Dx₁(v₁(t,x₁,x₂,θ)) + Dx₂(v₂(t,x₁,x₂,θ)) ) ~ 0
   ]

   # Initial and boundary conditions
   boundconds = [
        # initial conditions at t = 0
       u₁(tspan.lower, x₁, x₂, θ) ~ 0,
       u₂(tspan.lower, x₁, x₂, θ) ~ 0,
       v₁(tspan.lower, x₁, x₂, θ) ~ 0,
       v₂(tspan.lower, x₁, x₂, θ) ~ 0,
      σ₁₁(tspan.lower, x₁, x₂, θ) ~ 0,
      σ₁₂(tspan.lower, x₁, x₂, θ) ~ 0,
      σ₂₂(tspan.lower, x₁, x₂, θ) ~ 0,
        p(tspan.lower, x₁, x₂, θ) ~ 0,
      # boundary conditions at x₁ = x₁⁻
       u₁(t, x₁span.lower, x₂, θ) ~ 0,
       u₂(t, x₁span.lower, x₂, θ) ~ 0,
       v₁(t, x₁span.lower, x₂, θ) ~ 0,
       v₂(t, x₁span.lower, x₂, θ) ~ 0,
      σ₁₁(t, x₁span.lower, x₂, θ) ~ 0,
      σ₁₂(t, x₁span.lower, x₂, θ) ~ 0,
      σ₂₂(t, x₁span.lower, x₂, θ) ~ 0,
        p(t, x₁span.lower, x₂, θ) ~ 0,
      # boundary conditions at x₁ = x₁⁺
       u₁(t, x₁span.upper, x₂, θ) ~ 0,
       u₂(t, x₁span.upper, x₂, θ) ~ 0,
       v₁(t, x₁span.upper, x₂, θ) ~ 0,
       v₂(t, x₁span.upper, x₂, θ) ~ 0,
      σ₁₁(t, x₁span.upper, x₂, θ) ~ 0,
      σ₁₂(t, x₁span.upper, x₂, θ) ~ 0,
      σ₂₂(t, x₁span.upper, x₂, θ) ~ 0,
        p(t, x₁span.upper, x₂, θ) ~ 0,
      # boundary conditions at x₂ = x₂⁻
      σ₁₂(t, x₁, x₂span.lower, θ) ~ 0,
      # boundary conditions at x₂ = x₂⁺
       u₁(t, x₁, x₂span.upper, θ) ~ 0,
       u₂(t, x₁, x₂span.upper, θ) ~ 0,
       v₁(t, x₁, x₂span.upper, θ) ~ 0,
       v₂(t, x₁, x₂span.upper, θ) ~ 0,
      σ₁₁(t, x₁, x₂span.upper, θ) ~ 0,
      σ₁₂(t, x₁, x₂span.upper, θ) ~ 0,
      σ₂₂(t, x₁, x₂span.upper, θ) ~ 0,
        p(t, x₁, x₂span.upper, θ) ~ 0
   ]
   if ρ₀ˡ == 0
      push!(boundconds, σ₂₂(t, x₁, x₂span.lower, θ) ~ -p(t, x₁, x₂span.lower, θ))
   else
      push!(boundconds, σ₂₂(t, x₁, x₂span.lower, θ) ~ 0, p(t, x₁, x₂span.lower, θ) ~ 0)
   end

   # Time and space domains
   domain = [t ∈ tspan, x₁ ∈ x₁span, x₂ ∈ x₂span]

   # Neural network
   nnet = FastChain(FastDense(3,  64, Flux.tanh),
                    FastDense(64, 64, Flux.selu),
                    FastDense(64, 64, Flux.σ),
                    FastDense(64, 8))

   discretization = PhysicsInformedNN([Δt, Δx₁, Δx₂], nnet; strategy)

   pde_system = PDESystem(equations, boundconds, domain, [t, x₁, x₂], [u₁, u₂, v₁, v₂, σ₁₁, σ₁₂, σ₂₂, p])

   problem = NeuralPDE.discretize(pde_system, discretization)
   optimizer = Flux.ADAM(0.01)
   callback = function (p, l)
       println("Current loss is: $l")
       return false
   end

   res = GalacticOptim.solve(problem, optimizer; progress=false, cb=callback, maxiters)

   ϕ = discretization.phi
   θopt = res.minimizer

   return function sol(t, x₁, x₂)
      ϕ([t, x₁, x₂], θopt)
   end
end

# specify constants
const K, α, µ, ρ₀, ρ₀ˢ, ρ₀ˡ = let
   ρ₀ᶠˢ = 1400
   ρ₀ᶠˡ = 997
   cp₁ = 2000
   cp₂ = 1300
   cs = 1300
   d₀ = 0.3

   ρ₀ˢ = (1 - d₀)ρ₀ᶠˢ
   ρ₀ˡ = d₀ * ρ₀ᶠˡ
   ρ₀ = ρ₀ˢ + ρ₀ˡ
   µ  = ρ₀ˢ * cs^2
   K  = (cp₁^2 + cp₂^2 - 8/3*ρ₀ˡ/ρ₀*cs^2 - √((cp₁^2 - cp₂^2)^2 - 64/9*cs^4*ρ₀ˡ*ρ₀ˢ/ρ₀^2) ) * ρ₀ * ρ₀ˢ/(2ρ₀ˡ)
   α₃ = (cp₁^2 + cp₂^2 - 8/3*ρ₀ˢ/ρ₀*cs^2 - √((cp₁^2 - cp₂^2)^2 - 64/9*cs^4*ρ₀ˡ*ρ₀ˢ/ρ₀^2) ) / (2ρ₀^2)
   α  = ρ₀ * α₃ + K/ρ₀^2

   K, α, µ, ρ₀, ρ₀ˢ, ρ₀ˡ
end

# specify force functions F₁ and F₂
F₁(t, x₁, x₂) = 100exp(-t^2 - x₁^2 - x₂^2)
F₂(t, x₁, x₂) = 1000exp(-t^2 - x₁^2 - x₂^2)

# specify variable intervals
tspan  = IntervalDomain(0.0, 60.0)
x₁span = IntervalDomain(-100.0, 100.0)
x₂span = IntervalDomain(0.0, 100.0)

# specify discretization
Δt  = 20.0
Δx₁ = 40.0
Δx₂ = 25.0

# find solution to the PDE by fitting the neural network to dynamics
sol = fitPDESystem!(K, α, µ, ρ₀, ρ₀ˢ, ρ₀ˡ,
                    F₁, F₂,
                    tspan, x₁span, x₂span,
                    Δt, Δx₁, Δx₂)

sol(tspan.lower, x₁span.lower, x₂span.lower)
