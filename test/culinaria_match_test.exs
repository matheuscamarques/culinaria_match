defmodule CulinariaMatch do
  @moduledoc """
  V1: Implementação Padrão com Vetores Densos.
  """

  @doc """
  Calcula a similaridade de cosseno entre dois vetores densos (listas).
  Retorna um valor entre -1 e 1.
  """
  def cosine_similarity(vec_a, vec_b) do
    dot_product =
      Enum.zip(vec_a, vec_b)
      |> Enum.map(fn {a, b} -> a * b end)
      |> Enum.sum()

    mag_a = magnitude(vec_a)
    mag_b = magnitude(vec_b)

    if mag_a > 0 and mag_b > 0, do: dot_product / (mag_a * mag_b), else: 0.0
  end

  def magnitude(vec) do
    vec
    |> Enum.map(&:math.pow(&1, 2))
    |> Enum.sum()
    |> :math.sqrt()
  end

  def euclidean_distance(vec_a, vec_b) do
    Enum.zip(vec_a, vec_b)
    |> Enum.map(fn {a, b} -> :math.pow(a - b, 2) end)
    |> Enum.sum()
    |> :math.sqrt()
  end

  def normalize_preferences(preferences, preset_data) do
    # Converte o mapa de preferências esparso em um vetor denso (lista)
    Enum.reduce(preferences, preset_data, fn {categoria_id, valor}, acc ->
      # Força float para precisão
      List.replace_at(acc, categoria_id, valor * 1.0)
    end)
  end
end

defmodule CulinariaSparseMatch do
  @moduledoc """
  V2: Implementação com Representação Esparsa (Mapas).
  """

  @doc """
  Calcula a Similaridade de Cosseno entre dois mapas de preferências (Sparse Vectors).
  A vantagem aqui é que ignoramos as categorias com as quais os usuários não interagiram.
  """
  def cosine_similarity(map_a, map_b) do
    # 1. Encontrar a interseção das chaves (categorias em comum)
    keys_a = Map.keys(map_a)
    keys_b = Map.keys(map_b)

    # 2. Produto escalar (dot product) apenas das categorias em comum
    dot_product =
      keys_a
      |> Enum.filter(&(&1 in keys_b))
      |> Enum.map(fn id -> map_a[id] * map_b[id] end)
      |> Enum.sum()

    # 3. As magnitudes precisam considerar todos os pesos do usuário individualmente
    mag_a = magnitude(map_a)
    mag_b = magnitude(map_b)

    if mag_a > 0 and mag_b > 0, do: dot_product / (mag_a * mag_b), else: 0.0
  end

  def magnitude(map_preferences) do
    map_preferences
    |> Map.values()
    |> Enum.map(&:math.pow(&1, 2))
    |> Enum.sum()
    |> :math.sqrt()
  end

  def euclidean_distance(map_a, map_b) do
    # Na distância euclidiana esparsa, precisamos considerar a união de todas as chaves.
    # Se uma chave existe em A mas não em B, o valor em B é tratado como 0.0.
    all_keys = (Map.keys(map_a) ++ Map.keys(map_b)) |> Enum.uniq()

    all_keys
    |> Enum.map(fn id ->
      val_a = Map.get(map_a, id, 0.0)
      val_b = Map.get(map_b, id, 0.0)
      :math.pow(val_a - val_b, 2)
    end)
    |> Enum.sum()
    |> :math.sqrt()
  end
end

defmodule CulinariaDataPipeline do
  @total_users 1_000_000
  @quantization_scale 127

  def build_match_profile(raw_preferences, global_frequencies) do
    raw_preferences
    |> prune_noise(0.05)
    |> apply_icf_weighting(global_frequencies)
    |> amplify_dealbreakers(1.5)
    |> scale_preferences()
    |> quantize_to_int8()
  end

  def prune_noise(preferences, threshold) do
    max_val = preferences |> Map.values() |> Enum.map(&abs/1) |> Enum.max(fn -> 0 end)

    if max_val == 0,
      do: %{},
      else: Map.reject(preferences, fn {_id, val} -> abs(val) / max_val < threshold end)
  end

  def apply_icf_weighting(preferences, global_frequencies) do
    Map.new(preferences, fn {id, val} ->
      frequency = Map.get(global_frequencies, id, 1.0)
      icf_weight = :math.log(@total_users / frequency)
      {id, val * icf_weight}
    end)
  end

  def amplify_dealbreakers(preferences, penalty_multiplier) do
    Map.new(preferences, fn {id, val} ->
      if val < 0, do: {id, val * penalty_multiplier}, else: {id, val}
    end)
  end

  def scale_preferences(preferences) when preferences == %{}, do: %{}

  def scale_preferences(preferences) do
    max_val = preferences |> Map.values() |> Enum.map(&abs/1) |> Enum.max(fn -> 0 end)

    if max_val == 0.0,
      do: %{},
      else: Map.new(preferences, fn {id, val} -> {id, val / max_val} end)
  end

  def quantize_to_int8(preferences) do
    preferences
    |> Enum.reduce(%{}, fn {id, val}, acc ->
      quantized_val = round(val * @quantization_scale)
      if quantized_val != 0, do: Map.put(acc, id, quantized_val), else: acc
    end)
  end
end

defmodule CulinariaTaxonomia do
  @moduledoc """
  Taxonomia completa de categorias culinárias (100 dimensões).
  """

  @dicionario %{
    # --- LANCHES E FAST FOOD (0-19) ---
    0 => "Hambúrguer",
    1 => "Pizza",
    2 => "Hot Dog / Cachorro Quente",
    3 => "Pastel",
    4 => "Salgados em Geral",
    5 => "Coxinha",
    6 => "Esfiha",
    7 => "Sanduíches e Baguetes",
    8 => "Tapioca",
    9 => "Crepe Salgado",
    10 => "Shawarma",
    11 => "Kebab",
    12 => "Tacos",
    13 => "Burritos",
    14 => "Nachos",
    15 => "Porções / Batata Frita",
    16 => "Frango Frito (Estilo Americano/Coreano)",
    17 => "Fish and Chips",
    18 => "Waffles Salgados",
    19 => "Pão de Queijo",

    # --- CULINÁRIA INTERNACIONAL (20-39) ---
    20 => "Italiana",
    21 => "Francesa",
    22 => "Alemã",
    23 => "Portuguesa",
    24 => "Espanhola",
    25 => "Mexicana",
    26 => "Peruana",
    27 => "Argentina",
    28 => "Árabe",
    29 => "Libanesa",
    30 => "Turca",
    31 => "Grega",
    32 => "Chinesa",
    33 => "Coreana",
    34 => "Tailandesa",
    35 => "Indiana",
    36 => "Etíope",
    37 => "Marroquina",
    38 => "Vietnamita",
    39 => "Americana / BBQ",

    # --- CULINÁRIA BRASILEIRA E REFEIÇÕES (40-54) ---
    40 => "Brasileira (Geral)",
    41 => "Mineira",
    42 => "Nordestina",
    43 => "Baiana",
    44 => "Gaúcha",
    45 => "Paraense / Norte",
    46 => "Capixaba",
    47 => "Feijoada",
    48 => "Moqueca",
    49 => "Churrascaria / Carnes",
    50 => "Galeteria",
    51 => "Peixes e Frutos do Mar",
    52 => "Comida de Boteco",
    53 => "Marmita / Prato Feito (PF)",
    54 => "Sopas e Caldos",

    # --- ESTILO DE VIDA E NICHOS (55-69) ---
    55 => "Vegetariana",
    56 => "Vegana",
    57 => "Fit / Saudável",
    58 => "Low Carb",
    59 => "Sem Glúten",
    60 => "Sem Lactose",
    61 => "Orgânica",
    62 => "Kosher",
    63 => "Halal",
    64 => "Poke Havaiano",
    65 => "Ceviche",
    66 => "Saladas",
    67 => "Bowls Funcionais",
    68 => "Comida Viva / Raw Food",
    69 => "Dieta Mediterrânea",

    # --- DOCES E SOBREMESAS (70-84) ---
    70 => "Sobremesas em Geral",
    71 => "Sorvetes",
    72 => "Açaí",
    73 => "Bolos e Tortas",
    74 => "Doces Finos / Casamento",
    75 => "Chocolateria",
    76 => "Biscoitos e Cookies",
    77 => "Donuts",
    78 => "Churros",
    79 => "Crepes Doces",
    80 => "Frutas / Salada de Frutas",
    81 => "Doces Árabes",
    82 => "Confeitaria Clássica",
    83 => "Gelato Italiano",
    84 => "Milkshakes",

    # --- BEBIDAS, EMPÓRIO E OUTROS (85-99) ---
    85 => "Cafeteria",
    86 => "Padaria",
    87 => "Sucos e Smoothies",
    88 => "Chás e Infusões",
    89 => "Cervejaria Artesanal",
    90 => "Vinhos e Espumantes",
    91 => "Adega / Destilados",
    92 => "Drinks e Coquetelaria",
    93 => "Empório / Delicatessen",
    94 => "Açougue Gourmet",
    95 => "Queijos e Frios",
    96 => "Produtos Artesanais / Geléias",
    97 => "Refeições Congeladas",
    98 => "Suplementação / Whey",
    99 => "Bebidas Não Alcoólicas / Refrigerantes",
    100 => "Carne Bovina",
    101 => "Carne Suína",
    102 => "Frango / Aves",
    103 => "Peixe Branco",
    104 => "Salmão",
    105 => "Camarão",
    106 => "Frutos do Mar (Lula/Polvo)",
    107 => "Bacon",
    108 => "Ovo",
    109 => "Linguiça / Embutidos",
    110 => "Carne Seca / Charque",
    111 => "Costela",
    112 => "Carne Moída",
    113 => "Peito de Peru",
    114 => "Presunto",
    115 => "Salsicha",
    116 => "Atum",
    117 => "Tofu",
    118 => "Seitan (Carne de Glúten)",
    119 => "Soja (PTS)",

    # --- VEGETAIS, LEGUMES E HORTALIÇAS (120-139) ---
    120 => "Cebola",
    121 => "Alho",
    122 => "Tomate",
    123 => "Alface",
    124 => "Rúcula",
    125 => "Cenoura",
    126 => "Brócolis",
    127 => "Couve-flor",
    128 => "Pimentão",
    129 => "Batata Inglesa",
    130 => "Batata Doce",
    131 => "Mandioca / Aipim",
    132 => "Abóbora",
    133 => "Abobrinha",
    134 => "Berinjela",
    135 => "Espinafre",
    136 => "Milho",
    137 => "Ervilha",
    138 => "Cogumelos (Champignon/Shitake)",
    139 => "Repolho",

    # --- CARBOIDRATOS E GRÃOS (140-159) ---
    140 => "Arroz Branco",
    141 => "Arroz Integral",
    142 => "Feijão Carioca",
    143 => "Feijão Preto",
    144 => "Lentilha",
    145 => "Grão de Bico",
    146 => "Macarrão / Massa Tradicional",
    147 => "Trigo / Farinha de Trigo",
    148 => "Aveia",
    149 => "Quinoa",
    150 => "Pão Tradicional",
    151 => "Massa de Pizza",
    152 => "Massa de Pastel",
    153 => "Farinha de Mandioca",
    154 => "Fubá / Farinha de Milho",
    155 => "Polenta",
    156 => "Tapioca (Goma)",
    157 => "Pão Árabe / Sírio",
    158 => "Massa Integral",
    159 => "Croutons / Torradas",

    # --- LATICÍNIOS E DERIVADOS (160-179) ---
    160 => "Queijo Mussarela",
    161 => "Queijo Parmesão",
    162 => "Queijo Prato",
    163 => "Queijo Cheddar",
    164 => "Gorgonzola / Roquefort",
    165 => "Provolone",
    166 => "Requeijão",
    167 => "Cream Cheese",
    168 => "Leite Integral",
    169 => "Leite Desnatado",
    170 => "Leite Vegetal (Amêndoa/Aveia)",
    171 => "Manteiga",
    172 => "Margarina",
    173 => "Iogurte Natural",
    174 => "Creme de Leite",
    175 => "Leite Condensado",
    176 => "Doce de Leite",
    177 => "Coalhada",
    178 => "Ricota",
    179 => "Cottage",

    # --- TEMPEROS, ERVAS E CONDIMENTOS (180-199) ---
    180 => "Sal",
    181 => "Pimenta do Reino",
    182 => "Pimenta Vermelha (Dedo de Moça)",
    183 => "Orégano",
    184 => "Manjericão",
    185 => "Salsa e Cebolinha (Cheiro Verde)",
    186 => "Coentro",
    187 => "Alecrim",
    188 => "Cominho",
    189 => "Páprica (Doce/Picante/Defumada)",
    190 => "Curry",
    191 => "Açafrão / Cúrcuma",
    192 => "Azeite de Oliva",
    193 => "Óleo (Soja/Girassol)",
    194 => "Vinagre",
    195 => "Molho Shoyu",
    196 => "Mostarda",
    197 => "Ketchup",
    198 => "Maionese",
    199 => "Molho Barbecue",

    # --- FRUTAS E OLEAGINOSAS (200-219) ---
    200 => "Limão",
    201 => "Laranja",
    202 => "Morango",
    203 => "Banana",
    204 => "Maçã",
    205 => "Abacaxi",
    206 => "Uva",
    207 => "Coco",
    208 => "Abacate",
    209 => "Manga",
    210 => "Maracujá",
    211 => "Amendoim",
    212 => "Castanha de Caju",
    213 => "Castanha do Pará",
    214 => "Nozes",
    215 => "Amêndoas",
    216 => "Gergelim",
    217 => "Uva Passa",
    218 => "Ameixa Seca",
    219 => "Damasco",

    # --- ADOÇANTES, BASE DE DOCES E EXTRAS (220-230) ---
    220 => "Açúcar Refinado",
    221 => "Açúcar Mascavo / Demerara",
    222 => "Mel",
    223 => "Adoçante / Stevia",
    224 => "Cacau em Pó",
    225 => "Chocolate ao Leite",
    226 => "Chocolate Amargo / Meio Amargo",
    227 => "Essência de Baunilha",
    228 => "Fermento",
    229 => "Caldo de Carne/Frango (Tablete)",
    230 => "Bicarbonato de Sódio",
    # --- CALOR DIRETO E FOGO (230-244) ---
    230 => "Grelhado na Brasa (Carvão/Lenha)",
    231 => "Defumação a Quente (Hot Smoking)",
    232 => "Defumação a Frio (Cold Smoking)",
    233 => "Chapa / Smash",
    234 => "Fritura por Imersão",
    235 => "Fritura em Ar (Air Fryer)",
    236 => "Salteado (Sauté)",
    237 => "Wok (Estilo Oriental)",
    238 => "Grelhado Elétrico / Gás",
    239 => "Assado no Forno (Convecção)",
    240 => "Forno a Lenha",
    241 => "Gratinado",
    242 => "Maçaricado",
    243 => "Rotisserie / Espeto Giratório",
    244 => "Flambado",

    # --- CALOR ÚMIDO E LENTO (245-259) ---
    245 => "Cozido no Vapor",
    246 => "Fervura (Boiling)",
    247 => "Escaldado (Blanching)",
    248 => "Prensado",
    249 => "Ensopado / Guisado",
    250 => "Braseado (Braising)",
    251 => "Confitado (Cozimento em Gordura)",
    252 => "Sous-vide (Baixa Temperatura a Vácuo)",
    253 => "Cozimento Lento (Slow Cooker)",
    254 => "Banho-maria",
    255 => "Redução (Ex: Molhos concentrados)",
    256 => "Cozido sob Pressão",
    257 => "Infusão a Quente",
    258 => "Escalfado (Poaching)",

    # --- PROCESSAMENTO FRIO E TÉCNICAS BIOQUÍMICAS (260-274) ---
    260 => "Cru (Raw)",
    261 => "Marinado",
    262 => "Fermentação Natural (Sourdough/Lactofermentação)",
    263 => "Cura (Dry Aged / Salga)",
    264 => "Maturação (Aged)",
    265 => "Prensado a Frio (Cold Pressed)",
    266 => "Desidratado / Liofilizado",
    267 => "Emulsificado (Ex: Maioneses caseiras)",
    268 => "Macerado",
    269 => "Batido / Frappé",
    270 => "Moído na Hora (Ex: Grãos de café)",
    271 => "Torra Clara",
    272 => "Torra Média",
    273 => "Torra Escura",
    274 => "Extração por Gravidade (Filtrados)",

    # --- TEXTURAS E FINALIZAÇÃO (275-285) ---
    275 => "Empanado (Panko/Farinha)",
    276 => "Caramelizado",
    277 => "Defumação com Madeira de Frutífera",
    278 => "Glaciado",
    279 => "Trinchado / Fatiado Fino",
    280 => "Desfiado",
    281 => "Esferificado (Gastronomia Molecular)",
    282 => "Aerado (Sifão/Espumas)",
    283 => "Purê / Cremoso",
    284 => "Crocante / Pururuca",
    285 => "Al Dente",

    # --- ESCALA DE PICÂNCIA (SCOVILLE SIMULADA) (285-294) ---
    285 => "Picância Nula (Doce/Suave)",
    286 => "Picância Leve (Apenas calor residual)",
    287 => "Picância Média (Nível Jalapeño)",
    288 => "Picância Alta (Nível Habanero)",
    289 => "Picância Extrema (Nível Carolina Reaper)",
    290 => "Picância Adstringente (Tipo Wasabi/Raiz Forte)",

    # --- PERFIS DE SABOR DOMINANTES (295-309) ---
    295 => "Perfil Umami Intenso",
    296 => "Perfil Ácido / Citrino",
    297 => "Perfil Amargo (High IBU / Café Forte)",
    298 => "Perfil Adocicado (Low Sugar)",
    299 => "Perfil Doce Extremo",
    300 => "Perfil Salgado / Mineral",
    301 => "Perfil Defumado (Smoky)",
    302 => "Perfil Terroso (Mushrooms/Raízes)",
    303 => "Perfil Herbáceo (Ervas Frescas)",
    304 => "Perfil Floral",
    305 => "Perfil Alcoólico Percebível",
    306 => "Perfil Gorduroso / Untuoso",
    307 => "Perfil Metálico / Sangue (Carnes mal passadas)",
    308 => "Perfil Fermentado / Ácido Lático",
    309 => "Perfil Picante/Especidado (Especiarias)",

    # --- TEXTURAS SENSORIAIS (310-324) ---
    310 => "Textura Crocante (Crunchy)",
    311 => "Textura Cremosa / Aveludada",
    312 => "Textura Gelatinosa / Viscosa",
    313 => "Textura Fibrosa",
    314 => "Textura Macia / Tenra",
    315 => "Textura Granulada",
    316 => "Textura Efervescente / Carbonatada",
    317 => "Textura Seca / Adstringente (Vinhos/Chás)",
    318 => "Textura Elástica (Queijos/Massas)",
    319 => "Textura Aerada / Leve",

    # --- TEMPERATURA E IMPACTO (325-335) ---
    325 => "Temperatura Escaldante",
    326 => "Temperatura Morna / Conforto",
    327 => "Temperatura Ambiente",
    328 => "Temperatura Gelada",
    329 => "Temperatura Congelante (Granizados/Sorvetes)",
    330 => "Aroma Persistente (Ex: Alho/Fritura)",
    331 => "Aroma Volátil / Sutil",
    332 => "Final de Boca Longo (Aftertaste)",
    333 => "Final de Boca Curto / Limpo",
    334 => "Cor Vibrante / Apelo Visual Forte",
    335 => "Aparência Rústica / Artesanal",

    # --- METAS DE COMPOSIÇÃO CORPORAL (356-370) ---
    356 => "Bulking / Superávit Calórico (Ganho de Massa)",
    357 => "Cutting / Déficit Calórico (Definição)",
    358 => "Manutenção de Peso",
    359 => "Densidade Proteica Alta (High Protein)",
    360 => "Densidade Energética Alta (High Calorie)",
    361 => "Volume Alimentar Alto (Low Calorie Density)",
    362 => "Cetogênica (VLCKD)",
    363 => "Paleolítica",
    364 => "Jejum Intermitente Friendly",
    365 => "Equilíbrio de Macronutrientes (IIFYM)",

    # --- PERFORMANCE E SUPORTE AO TREINO (371-385) ---
    371 => "Pré-Treino (Energia Rápida / Baixa Fibra)",
    372 => "Pós-Treino (Recuperação Glicogênica)",
    373 => "Refeição de Absorção Lenta (Baixo IG)",
    374 => "Refeição de Absorção Rápida (Alto IG)",
    375 => "Reposição de Eletrólitos / Sais",
    376 => "Anti-Inflamatório Natural",
    377 => "Foco em Aminoácidos Essenciais",
    378 => "Suporte à Síntese de Proteína",
    379 => "Energia Sustentada (Longa Duração)",
    380 => "Auxílio na Digestão (Enzimático)",

    # --- RESTRIÇÕES E SENSIBILIDADES TÉCNICAS (386-400) ---
    386 => "Índice Glicêmico Baixo",
    387 => "Zero Açúcares Adicionados",
    388 => "Livre de Gorduras Trans",
    389 => "Baixo Teor de Sódio",
    390 => "Rico em Fibras",
    391 => "Rico em Micronutrientes (Vitaminas/Minerais)",
    392 => "Probiótico / Saúde Intestinal",
    393 => "Livre de Conservantes Artificiais",
    394 => "Alcalinizante",
    395 => "Antioxidante Alto",

    # --- PERFIL DE INGESTÃO (401-410) ---
    401 => "Refeição Líquida / Shake",
    402 => "Refeição Sólida Densa",
    403 => "Lanche de Bolso / Snack de Treino",
    404 => "Substituto de Refeição (MRP)",
    405 => "Hidratação Hipertônica",
    406 => "Estímulo Metabólico (Termogênico)",

    # --- ENTRETENIMENTO E VIBE (411-425) ---
    411 => "Música ao Vivo (Geral)",
    412 => "Música ao Vivo (Rock/Blues)",
    413 => "Música ao Vivo (Samba/Pagode)",
    414 => "Música Eletrônica / DJ Set",
    415 => "Stand-up Comedy / Show de Humor",
    416 => "Transmissão de Eventos Esportivos",
    417 => "Ambiente Badalado (Agito/High Volume)",
    418 => "Ambiente Intimista / Romântico",
    419 => "Ambiente Silencioso / Foco (Work-friendly)",
    420 => "Decoração Temática / Imersiva",
    421 => "Iluminação Baixa (Meia-luz)",
    422 => "Espaço Kids / Playground",
    423 => "Pet Friendly (Aceita Animais)",
    424 => "Rooftop / Vista Panorâmica",
    425 => "Área Externa / Deck ao Ar Livre",

    # --- SERVIÇO E VELOCIDADE (426-440) ---
    426 => "Fast Casual (Rápido mas de qualidade)",
    427 => "Fine Dining (Experiência de Luxo)",
    428 => "Autoatendimento / Totem",
    429 => "Serviço de Garçom Tradicional",
    430 => "Buffet Livre / Rodízio",
    431 => "Buffet por Quilo",
    432 => "Take-away (Retirada no Balcão)",
    433 => "Drive-thru",
    434 => "Cozinha Aberta (Show Cooking)",
    435 => "Speakeasy (Bar Secreto)",
    436 => "Atendimento Ultra-rápido (Executivo)",
    437 => "Menu Degustação",
    438 => "Happy Hour (Promoções de Bebidas)",
    439 => "Cozinha 24 Horas",
    440 => "Local de Fácil Estacionamento",

    # --- PÚBLICO E OCASIÃO (441-455) ---
    441 => "Ideal para Grupos Grandes",
    442 => "Ideal para Encontros (Date)",
    443 => "Ideal para Reuniões de Negócios",
    444 => "Ideal para Famílias com Crianças",
    445 => "Público Jovem / Universitário",
    446 => "Público Executivo / Corporativo",
    447 => "Público Alternativo / Underground",
    448 => "Ambiente Instagramável (Visual focado em fotos)",
    449 => "Experiência de Praia / Pé na Areia",
    450 => "Experiência Rústica / Campo",
    451 => "Lugar Histórico / Tradicional da Cidade",
    452 => "Vibe de 'Esquenta' (Pré-festa)",
    453 => "Vibe de 'Saideira' (Pós-festa)",
    454 => "Ambiente Minimalista / Moderno",
    455 => "Ambiente Acolhedor / 'Casa de Vó'",

    # --- RITUAIS DA MADRUGADA E MANHÃ (460-474) ---
    460 => "Madrugada Ativa (04:00 - 06:00)",
    461 => "Café da Manhã Precoce (Early Bird)",
    462 => "Café da Manhã Tradicional",
    463 => "Brunch (Final de semana)",
    464 => "Pico de Cafeína / Foco Matinal",
    465 => "Pré-Treino Matinal",
    466 => "Desjejum Proteico",

    # --- BLOCO DE MEIO-DIA E TARDE (475-489) ---
    475 => "Almoço Executivo (Rápido/Semana)",
    476 => "Almoço de Lazer (Longo/Fim de Semana)",
    477 => "Janela de Almoço Sagrada (12:00 em ponto)",
    478 => "Lanche da Tarde / Coffee Break",
    479 => "Pós-Treino Vespertino",
    480 => "Happy Hour Precoce",
    481 => "Chá da Tarde / Momento Relax",

    # --- NOITE E MADRUGADA TARDIA (490-504) ---
    490 => "Jantar de Rotina (Semana)",
    491 => "Jantar Festivo / Social (Fim de Semana)",
    492 => "Ceia Pré-Sono",
    493 => "Late Night Snack (Lanche de Madrugada)",
    494 => "Saideira de Balada / Pós-Evento",
    495 => "Jantar 'Comfort Food' (Descompressão)",

    # --- CICLOS SEMANAIS E SAZONAIS (505-520) ---
    505 => "Segunda-feira Detox / Foco",
    506 => "Sexta-feira 'Cheat Meal' / Liberação",
    507 => "Sábado Social / Exploração",
    508 => "Domingo em Família / Tradicional",
    509 => "Sazonalidade de Verão (Refrescante)",
    510 => "Sazonalidade de Inverno (Calórico/Quente)",
    511 => "Datas Comemorativas (Aniversários/Eventos)",
    512 => "Período de Férias / Viagem"
  }

  # O Elixir extrai as chaves do mapa em tempo de compilação e cria a lista
  # @categorias será um array real de [0, 1, 2, ..., 99]
  @categorias Map.keys(@dicionario)

  @doc "Retorna a lista completa de IDs (0 a 99)"
  def ids, do: @categorias

  @doc "Retorna o nome da categoria dado o seu ID"
  def nome(id), do: Map.get(@dicionario, id, "Desconhecido")

  @doc "Retorna o dicionário completo para debug ou APIs"
  def dicionario, do: @dicionario
end

defmodule CulinariaDatasetGenerator do
  @moduledoc """
  Gera um dataset sintético de usuários focado puramente em Categorias de Comida.
  Atribui categorias favoritas e odiadas, adicionando ruído para testar o pipeline.
  """

  # Agora as categorias são puxadas dinamicamente (0 a 99)
  @categorias CulinariaTaxonomia.ids()

  @doc """
  Gera 'n' usuários com preferências cruas.
  """
  def generate_users(num_users) do
    1..num_users
    |> Enum.map(fn id ->
      # Escolhe uma categoria favorita principal
      categoria_favorita = Enum.random(@categorias)

      # Escolhe uma categoria odiada (diferente da favorita)
      categoria_odiada =
        @categorias
        |> Enum.reject(&(&1 == categoria_favorita))
        |> Enum.random()

      # 1. Gera o gosto e aversão principais
      base_likes = %{
        categoria_favorita => Enum.random(300..600),
        categoria_odiada => Enum.random(-500..-200)
      }

      # 2. Adiciona ruído em outras categorias que o usuário "esbarrou"
      noise = generate_noise(categoria_favorita, categoria_odiada)

      # 3. Mescla tudo e simula o volume de uso
      raw_preferences =
        Map.merge(base_likes, noise, fn _k, v1, v2 -> v1 + v2 end)
        |> simulate_usage_volume()

      %{
        id: id,
        categoria_favorita: categoria_favorita,
        categoria_odiada: categoria_odiada,
        raw_preferences: raw_preferences
      }
    end)
  end

  def calculate_global_frequencies(users) do
    users
    |> Enum.flat_map(fn user -> Map.keys(user.raw_preferences) end)
    |> Enum.frequencies()
  end

  defp generate_noise(fav, hate) do
    @categorias
    |> Enum.reject(&(&1 in [fav, hate]))
    # MUDANÇA IMPORTANTE AQUI:
    # Como agora temos 100 categorias, o ruído humano aumentou.
    # O usuário agora esbarra aleatoriamente de 5 a 15 categorias!
    |> Enum.take_random(Enum.random(5..15))
    |> Map.new(fn category_id ->
      {category_id, Enum.random(-50..100)}
    end)
  end

  defp simulate_usage_volume(preferences) do
    volume_multiplier = Enum.random(1..20)

    Map.new(preferences, fn {id, val} ->
      {id, val * volume_multiplier}
    end)
  end
end

defmodule CulinariaScaleMatch do
  @moduledoc """
  V4: Normalização Escalar.
  """

  def scale_preferences(map_preferences) when map_preferences == %{}, do: %{}

  def scale_preferences(map_preferences) do
    max_val =
      map_preferences
      |> Map.values()
      |> Enum.map(&abs/1)
      |> Enum.max()

    case max_val do
      0 -> %{}
      0.0 -> %{}
      _ -> Map.new(map_preferences, fn {id, val} -> {id, val / max_val} end)
    end
  end
end

defmodule CulinariaSparseQuantized do
  @moduledoc """
  V5: Implementação com Quantização (Int8).
  """

  @scale_factor 127

  def quantize(map_normalized) do
    map_normalized
    |> Enum.reduce(%{}, fn {id, val}, acc ->
      quantized_val = round(val * @scale_factor)

      if quantized_val != 0 do
        Map.put(acc, id, quantized_val)
      else
        acc
      end
    end)
  end
end

defmodule CulinariaMatchTest do
  use ExUnit.Case, async: false

  test "Matching V1: Deve calcular o matching usando Vetores Densos" do
    preset_data = [0.0, 0.0, 0.0]

    user_a_raw = %{"nome" => "Alice", "preferencias" => %{0 => 5000, 1 => -2000}}
    user_b_raw = %{"nome" => "Bob", "preferencias" => %{0 => 5000, 1 => -2000, 2 => 1000}}

    vec_a = CulinariaMatch.normalize_preferences(user_a_raw["preferencias"], preset_data)
    vec_b = CulinariaMatch.normalize_preferences(user_b_raw["preferencias"], preset_data)

    distancia = CulinariaMatch.euclidean_distance(vec_a, vec_b)
    similaridade = CulinariaMatch.cosine_similarity(vec_a, vec_b)

    IO.puts("\n--- Matching V1 (Dense Vectors) ---")
    IO.puts("Resultados para #{user_a_raw["nome"]} e #{user_b_raw["nome"]}:")
    IO.puts("Distância Euclidiana: #{Float.round(distancia, 2)}")
    IO.puts("Similaridade Cosseno: #{Float.round(similaridade, 4)}")

    assert similaridade > 0.8
    assert is_float(distancia)
  end

  test "Matching V2: Deve calcular similaridade usando Representação Esparsa" do
    user_a = %{"nome" => "Alice", "preferencias" => %{0 => 5000, 1 => -2000}}
    user_b = %{"nome" => "Bob", "preferencias" => %{0 => 5000, 1 => -2000, 2 => 1000}}

    similaridade =
      CulinariaSparseMatch.cosine_similarity(user_a["preferencias"], user_b["preferencias"])

    distancia =
      CulinariaSparseMatch.euclidean_distance(user_a["preferencias"], user_b["preferencias"])

    IO.puts("\n--- Matching V2 (Sparse Vectors) ---")
    IO.puts("Usuários: #{user_a["nome"]} & #{user_b["nome"]}")
    IO.puts("Distância Euclidiana: #{Float.round(distancia, 2)}")
    IO.puts("Similaridade Cosseno: #{Float.round(similaridade, 4)}")

    assert similaridade > 0.8
    assert distancia > 0
  end

  test "Matching V3: Deve aplicar o Contextual Weighting Pipeline" do
    # O 1 é ruído
    user_a = %{"nome" => "Alice", "preferencias" => %{0 => 5000, 1 => -2000, 2 => 1}}
    user_b = %{"nome" => "Bob", "preferencias" => %{0 => 5000, 1 => -2000}}

    global_frequencies = %{0 => 900_000, 1 => 5_000, 2 => 100_000}

    # Pipeline da V3 usando as funções corretas do CulinariaDataPipeline
    processed_a =
      user_a["preferencias"]
      |> CulinariaDataPipeline.prune_noise(0.05)
      |> CulinariaDataPipeline.apply_icf_weighting(global_frequencies)
      |> CulinariaDataPipeline.amplify_dealbreakers(1.5)

    processed_b =
      user_b["preferencias"]
      |> CulinariaDataPipeline.prune_noise(0.05)
      |> CulinariaDataPipeline.apply_icf_weighting(global_frequencies)
      |> CulinariaDataPipeline.amplify_dealbreakers(1.5)

    similaridade = CulinariaSparseMatch.cosine_similarity(processed_a, processed_b)
    distancia = CulinariaSparseMatch.euclidean_distance(processed_a, processed_b)

    IO.puts("\n--- Matching V3 (Contextual Weighting Pipeline) ---")
    IO.puts("Usuários: #{user_a["nome"]} & #{user_b["nome"]}")
    IO.puts("Distância Euclidiana: #{Float.round(distancia, 2)}")
    IO.puts("Similaridade Cosseno: #{Float.round(similaridade, 4)}")

    assert similaridade > 0.8
    assert distancia >= 0.0
  end

  test "Matching V4: Deve normalizar pesos e calcular afinidade" do
    user_a = %{"nome" => "Alice", "preferencias" => %{0 => 5000, 1 => -2000}}
    user_b = %{"nome" => "Bob", "preferencias" => %{0 => 5000, 1 => -2000, 2 => 1000}}

    map_a = CulinariaScaleMatch.scale_preferences(user_a["preferencias"])
    map_b = CulinariaScaleMatch.scale_preferences(user_b["preferencias"])

    similaridade = CulinariaSparseMatch.cosine_similarity(map_a, map_b)
    distancia = CulinariaSparseMatch.euclidean_distance(map_a, map_b)

    IO.puts("\n--- Matching V4 (Normalization) ---")
    IO.puts("Usuários: #{user_a["nome"]} & #{user_b["nome"]}")
    IO.puts("Distância Euclidiana: #{Float.round(distancia, 2)}")
    IO.puts("Similaridade Cosseno: #{Float.round(similaridade, 4)}")

    assert similaridade > 0.8
    assert distancia > 0
  end

  test "Matching V5: Deve quantizar e manter a precisão do match" do
    user_a = %{"nome" => "Alice", "preferencias" => %{0 => 5000, 1 => -2000}}
    user_b = %{"nome" => "Bob", "preferencias" => %{0 => 5000, 1 => -2000, 2 => 1000}}

    normalized_map_a = CulinariaScaleMatch.scale_preferences(user_a["preferencias"])
    normalized_map_b = CulinariaScaleMatch.scale_preferences(user_b["preferencias"])

    quant_map_a = CulinariaSparseQuantized.quantize(normalized_map_a)
    quant_map_b = CulinariaSparseQuantized.quantize(normalized_map_b)

    similaridade = CulinariaSparseMatch.cosine_similarity(quant_map_a, quant_map_b)
    distancia = CulinariaSparseMatch.euclidean_distance(quant_map_a, quant_map_b)

    IO.puts("\n--- Matching V5 (Quantization) ---")
    IO.puts("Usuários: #{user_a["nome"]} & #{user_b["nome"]}")
    IO.puts("Distância Euclidiana: #{Float.round(distancia, 2)}")
    IO.puts("Similaridade Cosseno: #{Float.round(similaridade, 4)}")

    assert similaridade > 0.8
    assert distancia > 0
  end

  test "Gerador: Deve criar um dataset focado em Categorias e com ruídos" do
    users = CulinariaDatasetGenerator.generate_users(100)
    assert length(users) == 100

    sample_user = List.first(users)

    assert Map.has_key?(sample_user, :id)
    assert Map.has_key?(sample_user, :categoria_favorita)
    assert Map.has_key?(sample_user, :raw_preferences)

    # O mapa de preferências tem que ter gerado algum dado
    assert map_size(sample_user.raw_preferences) > 0
  end

  test "Integração: Pipeline deve processar dados caóticos e gerar Int8 limpo" do
    users = CulinariaDatasetGenerator.generate_users(100)
    global_frequencies = CulinariaDatasetGenerator.calculate_global_frequencies(users)

    user = List.first(users)

    processed_profile =
      CulinariaDataPipeline.build_match_profile(
        user.raw_preferences,
        global_frequencies
      )

    IO.puts("\n--- Integração End-to-End: Gerador -> Pipeline ---")
    IO.puts("Categoria Favorita Original: #{user.categoria_favorita}")
    IO.puts("Vetor Bruto (Sujo): #{inspect(user.raw_preferences)}")
    IO.puts("Vetor Quantizado (Limpo): #{inspect(processed_profile)}")

    assert is_map(processed_profile)

    # Regra de Ouro do Denoising
    assert map_size(processed_profile) <= map_size(user.raw_preferences)

    # Regra da Quantização (Int8)
    Enum.each(processed_profile, fn {_category_id, val} ->
      assert is_integer(val), "Falha na Quantização: O valor #{val} não é um inteiro"
      assert val >= -127 and val <= 127, "Estouro de Escala: O valor #{val} saiu do limite Int8"
      assert val != 0, "Denoising Falhou: Valores zero não deveriam existir no mapa esparso"
    end)
  end

  @tag timeout: :infinity
  test "Cenário Real: Matheus encontrando o seu grupo na base de dados" do
    # 1. Perfil do Matheus (2: Japonesa, 1: Pizza)
    matheus = %{
      id: 1,
      nome: "Matheus",
      categoria_favorita: 2,
      raw_preferences: %{
        2 => 1000,
        1 => -800,
        0 => 50
      }
    }

    # 2. Geramos o grupo (3.000 pessoas)
    grupo = CulinariaDatasetGenerator.generate_users(100)
    todos_usuarios = [matheus | grupo]
    global_frequencies = CulinariaDatasetGenerator.calculate_global_frequencies(todos_usuarios)

    # 3. Processamento do Perfil
    matheus_profile =
      CulinariaDataPipeline.build_match_profile(
        matheus.raw_preferences,
        global_frequencies
      )

    # 4. Cálculo de Matches
    matches =
      grupo

      |> Task.async_stream(
        fn pessoa ->
          # O processamento pesado acontece aqui dentro, em processos separados
          pessoa_profile =
            CulinariaDataPipeline.build_match_profile(
              pessoa.raw_preferences,
              global_frequencies
            )

          similaridade = CulinariaSparseMatch.cosine_similarity(matheus_profile, pessoa_profile)

          %{
            id: pessoa.id,
            categoria_favorita: pessoa.categoria_favorita,
            categoria_odiada: pessoa.categoria_odiada,
            similaridade: similaridade
          }
        end,
        # Usa todos os cores disponíveis
        max_concurrency: System.schedulers_online(),
        # Ajuste conforme a carga
        timeout: 15_000
      )
      # Desembrulha o retorno da Task
      |> Enum.map(fn {:ok, result} -> result end)
      |> Enum.sort_by(& &1.similaridade, :desc)

    top_10 = Enum.take(matches, 10)

    # --- CONFIGURAÇÃO DA TABELA ---
    w_pos = 5
    w_id = 10
    w_cat = 40
    w_odiada = 40
    w_afin = 10

    linha =
      "+-#{String.duplicate("-", w_pos)}-+-#{String.duplicate("-", w_id)}-+-#{String.duplicate("-", w_cat)}-+-#{String.duplicate("-", w_odiada)}-+-#{String.duplicate("-", w_afin)}-+"

    IO.puts("\n--- Motor de Matchmaking: Matheus vs O Mundo ---")

    IO.puts(
      "Perfil do Matheus: Ama [#{CulinariaTaxonomia.nome(2)}], Odeia [#{CulinariaTaxonomia.nome(1)}]"
    )

    IO.puts("\nTop 10 Matches (Afinidade em Inteiros):")

    IO.puts(linha)

    IO.puts(
      "| #{String.pad_trailing("POS", w_pos)} | #{String.pad_trailing("ID PESSOA", w_id)} | #{String.pad_trailing("CATEGORIA FAVORITA", w_cat)} | #{String.pad_trailing("CATEGORIA ODIADA", w_odiada)} | #{String.pad_trailing("AFINIDADE", w_afin)} |"
    )

    IO.puts(linha)

    # Função auxiliar para truncar nomes longos
    format_cell = fn id, width ->
      nome = "#{id} - #{CulinariaTaxonomia.nome(id)}"

      if String.length(nome) > width do
        String.slice(nome, 0, width - 3) <> "..."
      else
        String.pad_trailing(nome, width)
      end
    end

    top_10
    |> Enum.with_index(1)
    |> Enum.each(fn {match, idx} ->
      pos_str = String.pad_trailing("#{idx}º", w_pos)
      id_str = String.pad_trailing("##{match.id}", w_id)

      cat_fav_str = format_cell.(match.categoria_favorita, w_cat)
      cat_odi_str = format_cell.(match.categoria_odiada, w_odiada)

      # Afinidade convertida para INTEIRO
      percentual_int = round(match.similaridade * 100)
      afin_str = String.pad_leading("#{percentual_int}%", w_afin)

      IO.puts("| #{pos_str} | #{id_str} | #{cat_fav_str} | #{cat_odi_str} | #{afin_str} |")

      # Assert para garantir qualidade mínima no topo
      # assert match.similaridade > 0.4
    end)

    IO.puts(linha <> "\n")
  end
end
