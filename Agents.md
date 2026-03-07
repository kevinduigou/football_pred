# AGENTS.md

## Code Style
- **Immutability First**: 
   - Treat all function parameters as **immutable**. Never modify inputs directly.
   - Prefer pure functions and immutable objects with `@final` and `__slots__`.
- **Make Illegal States Unrepresentable**:
  - Domain objects must not allow invalid construction.
  - Use factory methods like `try_create()` or validation before instantiation.
  - Never expose public constructors for domain objects with constraints.
- **Be Explicit**: No dynamic behavior or magic (e.g., metaclasses, monkey patching, decorators that inject logic).
- **Explicit**: Function and variable names shall be as explicit as possible
- **Rust-style Control Flow**:
  - Use `try`, `except`, or `raise` only in the infrastructure layer to open file, send requests, ...
  - All fallible functions must return a `Result[T, E]` using the returns library (already installed in pyproject.toml).
- Domain objects shall be simple dataclass using frozen true for value object
- Do not allow domain objects to depend on external libraries, ORMs, or I/O.

- When doing structural modification to the application, use clear separation of concern between the layers:
  - **Domain**: Core business logic. No DB, I/O, or HTTP code.
  - **Application**: Use cases, orchestration. Ports. Stateless.
  - **Infrastructure**: Implements ports using DBs, APIs, or I/O.
  - **Interface**: Fastapi interface.
- Define **ports (interfaces)** in the application layer.
- Implement ports (adapters) in infrastructure.
- Application Layer is the home of use cases, prefer uses cases over too general-purpose services.

## Buildng and Testing

- Once a modification is done on Python files, always perform 
uv run pytest then if it ok follow with
uv run ruff check ./src/ then if it ok follow with
uv run mypy ./src/ then if it ok follow with

and finish by
uv run black ./


