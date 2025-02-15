# scripts/gui/ecm_tab.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import jax
import jax.numpy as jnp
import optax
from scipy.io import loadmat
import collimator
from collimator.library import LookupTable1d
from collimator.simulation import SimulatorOptions

class ECMTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Inicializando ECMTab Completo")
        self.create_widgets()
        self.diagram = None
        self.context = None
        self.opt_params = None

    def create_widgets(self):
        self.logger.debug("Creando widgets completos de ECMTab")
        try:
            # Sección para seleccionar archivos .mat
            file_frame = ttk.LabelFrame(self, text="Selección de Archivos .mat")
            file_frame.pack(fill='x', padx=10, pady=5)

            ttk.Label(file_frame, text="Archivo de Entrenamiento:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
            self.train_file_entry = ttk.Entry(file_frame, width=50)
            self.train_file_entry.grid(row=0, column=1, padx=5, pady=5)
            ttk.Button(file_frame, text="Buscar", command=self.browse_train_file).grid(row=0, column=2, padx=5, pady=5)

            ttk.Label(file_frame, text="Archivo de Validación:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
            self.val_file_entry = ttk.Entry(file_frame, width=50)
            self.val_file_entry.grid(row=1, column=1, padx=5, pady=5)
            ttk.Button(file_frame, text="Buscar", command=self.browse_val_file).grid(row=1, column=2, padx=5, pady=5)

            # Sección para definir parámetros del modelo
            params_frame = ttk.LabelFrame(self, text="Parámetros del Modelo ECM")
            params_frame.pack(fill='x', padx=10, pady=5)

            self.params_entries = {}
            params = ['v0(s)', 'Rs(s)', 'R1(s)', 'C1(s)']
            for idx, param in enumerate(params):
                ttk.Label(params_frame, text=f"{param}:").grid(row=idx, column=0, padx=5, pady=5, sticky='e')
                entry = ttk.Entry(params_frame, width=50)  # Aumentar el ancho para facilitar la entrada
                entry.grid(row=idx, column=1, padx=5, pady=5, columnspan=2, sticky='w')
                self.params_entries[param] = entry

            # Sección de botones para ejecutar pasos
            buttons_frame = ttk.Frame(self)
            buttons_frame.pack(fill='x', padx=10, pady=5)

            self.init_button = ttk.Button(buttons_frame, text="Inicializar Modelo", command=self.initialize_model)
            self.init_button.pack(side='left', padx=5, pady=5)

            self.simulate_button = ttk.Button(buttons_frame, text="Simular", command=self.simulate_model, state='disabled')
            self.simulate_button.pack(side='left', padx=5, pady=5)

            self.optimize_button = ttk.Button(buttons_frame, text="Optimizar", command=self.optimize_model, state='disabled')
            self.optimize_button.pack(side='left', padx=5, pady=5)

            # Área para mostrar gráficos
            plot_frame = ttk.LabelFrame(self, text="Resultados de la Simulación")
            plot_frame.pack(fill='both', expand=True, padx=10, pady=5)

            self.figure = plt.Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            self.logger.debug("Widgets completos creados exitosamente en ECMTab")
        except Exception as e:
            self.logger.error(f"Error al crear widgets completos: {e}")
            tk.messagebox.showerror("Error", f"Fallo al crear widgets completos: {e}")

    def browse_train_file(self):
        self.logger.debug("Navegando para seleccionar archivo de entrenamiento")
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            self.train_file_entry.delete(0, tk.END)
            self.train_file_entry.insert(0, file_path)
            self.logger.info(f"Archivo de entrenamiento seleccionado: {file_path}")

    def browse_val_file(self):
        self.logger.debug("Navegando para seleccionar archivo de validación")
        file_path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat")])
        if file_path:
            self.val_file_entry.delete(0, tk.END)
            self.val_file_entry.insert(0, file_path)
            self.logger.info(f"Archivo de validación seleccionado: {file_path}")

    def initialize_model(self):
        self.logger.debug("Inicializando modelo ECM")
        train_file = self.train_file_entry.get()
        val_file = self.val_file_entry.get()
        if not train_file or not val_file:
            messagebox.showerror("Error", "Por favor, seleccione ambos archivos .mat.")
            self.logger.error("Archivos .mat no seleccionados")
            return

        try:
            # Extraer datos de los archivos .mat
            t_train, vt_train, curr_train, soc_train = self.extract_features_from_matfile(train_file)
            t_val, vt_val, curr_val, soc_val = self.extract_features_from_matfile(val_file)

            self.logger.info("Datos extraídos exitosamente de los archivos .mat")

            # Guardar datos para uso posterior
            self.t_train = t_train
            self.vt_train = vt_train
            self.curr_train = curr_train
            self.soc_train = soc_train

            self.t_val = t_val
            self.vt_val = vt_val
            self.curr_val = curr_val
            self.soc_val = soc_val

            # Habilitar el botón de simular
            self.simulate_button.config(state='normal')
            messagebox.showinfo("Éxito", "Modelo inicializado correctamente.")
            self.logger.info("Modelo inicializado y listo para simular.")
        except Exception as e:
            self.logger.error(f"Error al inicializar el modelo: {e}")
            messagebox.showerror("Error", f"Fallo al inicializar el modelo: {e}")

    def extract_features_from_matfile(self, filename, Q=3.0):
        self.logger.debug(f"Extrayendo características del archivo {filename}")
        data = loadmat(filename)
        t = data["meas"][0][0][0]
        vt = data["meas"][0][0][2]
        curr = -data["meas"][0][0][3]
        D = -data["meas"][0][0][4]
        soc = (Q - D) / Q
        return (
            jnp.array(t[:, 0]),
            jnp.array(vt[:, 0]),
            jnp.array(curr[:, 0]),
            jnp.array(soc[:, 0]),
        )

    def simulate_model(self):
        self.logger.debug("Simulando modelo ECM")
        try:
            # Obtener parámetros del modelo desde las entradas
            v0_points = self.get_model_params('v0(s)')
            Rs_points = self.get_model_params('Rs(s)')
            R1_points = self.get_model_params('R1(s)')
            C1_points = self.get_model_params('C1(s)')

            # Crear el sistema
            self.diagram, self.context = self.make_system(
                self.t_train, self.curr_train, self.vt_train,
                v0_points, Rs_points, R1_points, C1_points
            )

            self.logger.info("Sistema ECM creado exitosamente.")

            # Ejecutar la simulación y plotear resultados
            fig = self.forward_plot(
                v0_points, Rs_points, R1_points, C1_points,
                self.diagram, self.context,
                t_sim=self.t_train[-1],
                t_exp=self.t_train,
                soc_exp=self.soc_train
            )
            self.display_plot(fig)

            # Habilitar el botón de optimizar
            self.optimize_button.config(state='normal')
            messagebox.showinfo("Éxito", "Simulación completada exitosamente.")
            self.logger.info("Simulación completada y resultados mostrados.")
        except Exception as e:
            self.logger.error(f"Error en la simulación: {e}")
            messagebox.showerror("Error", f"Fallo en la simulación: {e}")

    def get_model_params(self, param_name):
        self.logger.debug(f"Obteniendo parámetros para {param_name}")
        entry = self.params_entries.get(param_name)
        if not entry:
            raise ValueError(f"Parámetro {param_name} no encontrado.")
        value_str = entry.get()
        if not value_str:
            raise ValueError(f"Por favor, ingrese el valor para {param_name}.")
        # Convertir la cadena de entrada a una lista de floats
        try:
            values = [float(x.strip()) for x in value_str.split(',')]
            self.logger.debug(f"Parámetros {param_name}: {values}")
            return jnp.array(values)
        except Exception as e:
            self.logger.error(f"Formato inválido para {param_name}: {e}")
            raise ValueError(f"Formato inválido para {param_name}. Use comas para separar los valores.")

    def make_system(self, t_exp_data, curr_exp_data, vt_exp_data, v0_points, Rs_points, R1_points, C1_points):
        self.logger.debug("Creando el sistema ECM con los parámetros proporcionados")
        Q = 3.0  # Capacidad de la batería
        builder = collimator.DiagramBuilder()

        # Crear los bloques del sistema
        discharge_current = builder.add(
            LookupTable1d(t_exp_data, curr_exp_data, "linear", name="discharge_current")
        )

        battery = builder.add(
            collimator.models.Battery(Q=Q, name="battery")
        )

        l2_loss = builder.add(self.make_l2_loss(name="l2_loss"))

        clock = builder.add(collimator.library.Clock(name="clock"))
        vt_exp = builder.add(
            LookupTable1d(t_exp_data, vt_exp_data, "linear", name="vt_exp")
        )

        # Conexiones
        builder.connect(clock.output_ports[0], discharge_current.input_ports[0])
        builder.connect(discharge_current.output_ports[0], battery.input_ports[0])
        builder.connect(clock.output_ports[0], vt_exp.input_ports[0])
        builder.connect(battery.output_ports[1], l2_loss.input_ports[0])
        builder.connect(vt_exp.output_ports[0], l2_loss.input_ports[1])

        # Exportar la salida del error L2
        builder.export_output(l2_loss.output_ports[0])

        diagram = builder.build()
        context = diagram.create_context()

        self.logger.debug("Sistema ECM creado correctamente")
        return diagram, context

    def make_l2_loss(self, name="l2_loss"):
        self.logger.debug(f"Creando bloque de pérdida L2: {name}")
        builder = collimator.DiagramBuilder()

        err = builder.add(collimator.library.Adder(2, operators="+-", name="err"))
        sq_err = builder.add(collimator.library.Power(2.0, name="sq_err"))
        sq_err_int = builder.add(collimator.library.Integrator(0.0, name="sq_err_int"))

        builder.connect(err.output_ports[0], sq_err.input_ports[0])
        builder.connect(sq_err.output_ports[0], sq_err_int.input_ports[0])

        # Exportar entradas y salidas
        builder.export_input(err.input_ports[0])
        builder.export_input(err.input_ports[1])
        builder.export_output(sq_err_int.output_ports[0])

        return builder.build(name=name)

    def forward_plot(self, v0_points, Rs_points, R1_points, C1_points, diagram, context, t_sim, t_exp, soc_exp, lw=0.5):
        self.logger.debug("Ejecutando simulación y creando gráficos")
        # Actualizar parámetros en el contexto
        new_params = {
            "v0_points": v0_points,
            "Rs_points": Rs_points,
            "R1_points": R1_points,
            "C1_points": C1_points,
        }

        subcontext = context[diagram["battery"].system_id].with_parameters(new_params)
        context = context.with_subcontext(diagram["battery"].system_id, subcontext)

        recorded_signals = {
            "vt": diagram["battery"].output_ports[1],
            "vt_exp": diagram["vt_exp"].output_ports[0],
            "soc": diagram["battery"].output_ports[0],
            "discharge_current": diagram["discharge_current"].output_ports[0],
            "l2_loss": diagram["l2_loss"].output_ports[0],
        }

        options = SimulatorOptions(
            max_major_steps=20,
        )

        sol = collimator.simulate(
            diagram,
            context,
            (0.0, t_sim),
            options=options,
            recorded_signals=recorded_signals,
        )
        l2_loss_final = sol.outputs["l2_loss"][-1]
        self.logger.info(f"L2 Loss final: {l2_loss_final}")

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(11, 8))
        fig.tight_layout()
        ax0.plot(
            sol.time,
            sol.outputs["discharge_current"],
            "-b",
            label="Corriente de descarga: sim",
            lw=lw,
        )
        ax1.plot(sol.time, sol.outputs["vt"], "-b", label="Voltaje terminal: sim", lw=lw)
        ax1.plot(
            sol.time, sol.outputs["vt_exp"], "-r", label="Voltaje terminal: exp", lw=lw
        )
        ax2.plot(sol.time, sol.outputs["soc"], "-b", label="SoC: sim", lw=lw)
        if (t_exp is not None) and (soc_exp is not None):
            ax2.plot(t_exp, soc_exp, "-r", label="SoC: exp", lw=lw)
        ax3.plot(sol.time, sol.outputs["l2_loss"], label=r"∫e²dt", lw=lw)
        ax0.legend()
        ax1.legend()
        ax2.legend()
        ax3.legend()
        fig.tight_layout()

        return fig

    def display_plot(self, fig):
        self.logger.debug("Mostrando gráfico en la interfaz")
        self.figure.clf()
        canvas = FigureCanvasTkAgg(fig, master=self.canvas.get_tk_widget())
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.figure = fig  # Actualizar la figura en el canvas
        self.logger.debug("Gráfico mostrado correctamente")

    def optimize_model(self):
        self.logger.debug("Iniciando optimización del modelo ECM")
        try:
            if not self.diagram or not self.context:
                raise ValueError("El modelo no ha sido inicializado o simulado aún.")

            # Configurar los parámetros de referencia
            v0_ref = 3.0
            Rs_ref = 15e-03
            R1_ref = 15e-03
            C1_ref = 3e03

            T_train = self.t_train[-1]

            # Definir la función de pérdida
            @jax.jit
            def forward(log_params, context):
                # Transformar los parámetros
                params = jnp.exp(log_params)
                params_arr = params.reshape((4, 11))

                new_params = {
                    "v0_points": v0_ref * params_arr[0, :],
                    "Rs_points": Rs_ref * params_arr[1, :],
                    "R1_points": R1_ref * params_arr[2, :],
                    "C1_points": C1_ref * params_arr[3, :],
                }

                subcontext = context[self.diagram["battery"].system_id].with_parameters(new_params)
                new_context = context.with_subcontext(self.diagram["battery"].system_id, subcontext)

                solver = "Tsit5"
                options = SimulatorOptions(
                    enable_autodiff=True, max_major_steps=20,
                )

                sol = collimator.simulate(self.diagram, new_context, (0.0, T_train), options=options)

                l2_loss = sol.context[self.diagram["l2_loss"].system_id].continuous_state

                cost = (1.0 / T_train) * l2_loss
                return cost

            # Valores iniciales
            v0_points_0 = self.get_model_params('v0(s)')
            Rs_points_0 = self.get_model_params('Rs(s)')
            R1_points_0 = self.get_model_params('R1(s)')
            C1_points_0 = self.get_model_params('C1(s)')

            log_params_0 = jnp.hstack(
                [
                    jnp.log(v0_points_0 / v0_ref),
                    jnp.log(Rs_points_0 / Rs_ref),
                    jnp.log(R1_points_0 / R1_ref),
                    jnp.log(C1_points_0 / C1_ref),
                ]
            )

            # Definir el optimizador
            optimizer = optax.adam(learning_rate=0.01)
            opt_state = optimizer.init(log_params_0)
            log_params = log_params_0

            # Calcular el gradiente
            grad_forward = jax.jit(jax.grad(forward))

            # Bucle de optimización
            num_epochs = 1000
            for epoch in range(num_epochs):
                gradients = grad_forward(log_params, self.context)
                updates, opt_state = optimizer.update(gradients, opt_state)
                log_params = optax.apply_updates(log_params, updates)

                if (epoch + 1) % 50 == 0:
                    current_loss_value = forward(log_params, self.context)
                    self.logger.info(
                        f"Epoch [{epoch+1}/{num_epochs}]: forward(log_params) = {current_loss_value}"
                    )
                    print(f"Epoch [{epoch+1}/{num_epochs}]: forward(log_params) = {current_loss_value}")

            # Guardar los parámetros optimizados
            opt_log_params = log_params
            opt_params = jnp.exp(opt_log_params)
            opt_params_arr = opt_params.reshape((4, 11))
            self.opt_params = {
                "v0_points": v0_ref * opt_params_arr[0, :],
                "Rs_points": Rs_ref * opt_params_arr[1, :],
                "R1_points": R1_ref * opt_params_arr[2, :],
                "C1_points": C1_ref * opt_params_arr[3, :],
            }

            self.logger.info("Optimización completada exitosamente.")

            # Ejecutar la simulación con parámetros optimizados y plotear resultados
            fig = self.forward_plot(
                v0_points=self.opt_params["v0_points"],
                Rs_points=self.opt_params["Rs_points"],
                R1_points=self.opt_params["R1_points"],
                C1_points=self.opt_params["C1_points"],
                diagram=self.diagram,
                context=self.context,
                t_sim=self.t_train[-1],
                t_exp=self.t_train,
                soc_exp=self.soc_train,
                lw=0.7
            )
            self.display_plot(fig)
            messagebox.showinfo("Éxito", "Optimización completada y resultados mostrados.")
            self.logger.info("Optimización completada y resultados mostrados.")
        except Exception as e:
            self.logger.error(f"Error en la optimización: {e}")
            messagebox.showerror("Error", f"Fallo en la optimización: {e}")
